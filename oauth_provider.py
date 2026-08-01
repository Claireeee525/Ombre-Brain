"""Persistent single-owner OAuth 2.1 provider for the remote Ombre MCP server.

The MCP Python SDK owns protocol validation, PKCE verification, discovery,
registration and token endpoints.  This module only supplies the durable
credential store and the small owner-consent page used by Ombre.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizeError,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


logger = logging.getLogger("ombre_brain.oauth")


def _secret_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class OmbreRefreshToken(RefreshToken):
    resource: str | None = None


class OmbreOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, OmbreRefreshToken, AccessToken]
):
    """OAuth provider with restart-safe storage and a private service-token lane."""

    def __init__(
        self,
        *,
        store_path: str,
        issuer_url: str,
        resource_url: str,
        verify_owner_password: Callable[[str], bool],
        service_token: str = "",
        scope: str = "ombre:memory",
        owner_name: str = "Claire",
        pending_ttl_seconds: int = 30 * 60,
    ) -> None:
        self.store_path = Path(store_path)
        self.issuer_url = issuer_url.rstrip("/")
        self.resource_url = resource_url.rstrip("/")
        self.verify_owner_password = verify_owner_password
        self.service_token = str(service_token or "")
        self.scope = scope
        self.owner_name = owner_name
        self.pending_ttl_seconds = max(10 * 60, int(pending_ttl_seconds))
        self._lock = asyncio.Lock()
        self._failed_logins: dict[str, deque[float]] = defaultdict(deque)
        self._store = self._read_store()

    @staticmethod
    def _empty_store() -> dict[str, Any]:
        return {
            "version": 1,
            "clients": {},
            "authorization_codes": {},
            "access_tokens": {},
            "refresh_tokens": {},
            "pending_authorizations": {},
            "audit": [],
        }

    def _read_store(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.store_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("OAuth store is not an object")
        except FileNotFoundError:
            parsed = self._empty_store()
        except Exception:
            # Never silently overwrite a malformed credential store.
            raise RuntimeError(f"Cannot read OAuth store: {self.store_path}")
        clean = self._empty_store()
        for key in clean:
            if key in parsed and isinstance(parsed[key], type(clean[key])):
                clean[key] = parsed[key]
        return clean

    def _reload_store(self) -> None:
        """Reload durable state so callbacks can land on another worker safely."""
        self._store = self._read_store()

    def _prune(self) -> None:
        now = time.time()
        for collection in ("authorization_codes", "access_tokens", "refresh_tokens"):
            records = self._store[collection]
            expired = [key for key, value in records.items() if value.get("expires_at") and value["expires_at"] < now]
            for key in expired:
                records.pop(key, None)
        pending = self._store["pending_authorizations"]
        for key in [key for key, value in pending.items() if value.get("expires_at", 0) < now]:
            pending.pop(key, None)
        self._store["audit"] = self._store["audit"][-500:]

    def _audit(self, event: str, **details: Any) -> None:
        safe = {key: value for key, value in details.items() if key not in {"token", "code", "password"}}
        self._store["audit"].append({"event": event, "at": int(time.time()), **safe})

    def _write_store(self) -> None:
        self._prune()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._store, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, self.store_path)

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        self._reload_store()
        record = self._store["clients"].get(str(client_id or ""))
        return OAuthClientInformationFull.model_validate(record) if record else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            raise ValueError("No client_id provided")
        async with self._lock:
            self._reload_store()
            self._store["clients"][client_info.client_id] = client_info.model_dump(mode="json")
            self._audit("client_registered", client_id=client_info.client_id, client_name=client_info.client_name or "")
            self._write_store()

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if params.resource and params.resource.rstrip("/") != self.resource_url:
            raise AuthorizeError(error="invalid_request", error_description="Resource does not match this Ombre server")
        # Use a server nonce for the password/consent page.  The client's OAuth
        # state is preserved separately for the final redirect, so a malicious
        # registered client cannot overwrite another pending login by choosing
        # the same state value.
        login_state = secrets.token_urlsafe(32)
        pending = {
            "client_id": client.client_id,
            "client_name": client.client_name or "MCP client",
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": bool(params.redirect_uri_provided_explicitly),
            "code_challenge": params.code_challenge,
            "scopes": params.scopes or [self.scope],
            "resource": params.resource or self.resource_url,
            "oauth_state": params.state,
            "expires_at": time.time() + self.pending_ttl_seconds,
        }
        async with self._lock:
            self._reload_store()
            self._store["pending_authorizations"][_secret_key(login_state)] = pending
            self._write_store()
        return f"{self.issuer_url}/oauth/login?{urlencode({'state': login_state})}"

    async def login_page(self, request: Request) -> Response:
        state = str(request.query_params.get("state") or "")
        self._reload_store()
        pending = self._store["pending_authorizations"].get(_secret_key(state))
        if not state or not pending or pending.get("expires_at", 0) < time.time():
            logger.warning("OAuth login state missing or expired (pending=%d)", len(self._store["pending_authorizations"]))
            return HTMLResponse("<h1>这次连接请求已失效，请回到官端重试。</h1>", status_code=400)
        client_name = html.escape(str(pending.get("client_name") or "Claude"))
        state_value = html.escape(state, quote=True)
        owner_name = html.escape(self.owner_name)
        callback_url = html.escape(f"{self.issuer_url}/oauth/callback", quote=True)
        script_nonce = secrets.token_urlsafe(18)
        body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>连接 Ombre</title><style>
body{{margin:0;background:#11151b;color:#eaf1f7;font:15px/1.6 -apple-system,BlinkMacSystemFont,sans-serif}}
main{{max-width:430px;margin:10vh auto;padding:28px;border:1px solid #334252;border-radius:20px;background:#18202a}}
h1{{font-size:23px;margin:0 0 8px}}p{{color:#aebdca}}label{{display:block;margin:22px 0 7px}}
input{{box-sizing:border-box;width:100%;padding:13px;border:1px solid #46596d;border-radius:10px;background:#0f141a;color:#fff}}
button{{width:100%;margin-top:18px;padding:13px;border:0;border-radius:12px;background:#86a9c5;color:#0e1720;font-weight:700}}
button:disabled{{opacity:.72;cursor:wait}}#submit-status{{min-height:24px;margin:12px 0 0;color:#b8cada}}
small{{display:block;margin-top:16px;color:#8093a5}}
</style></head><body><main><h1>把 Ombre 连接给 {client_name}</h1>
<p>登录后，官端可以读取和使用 {owner_name} 的 Ombre 记忆工具。没有确认就不会放行。</p>
<form id="oauth-form" method="post" action="{callback_url}"><input type="hidden" name="state" value="{state_value}">
<label for="password">Ombre 密码</label><input id="password" name="password" type="password" autocomplete="current-password" required autofocus>
<button id="connect-button" type="submit">确认连接</button><p id="submit-status" role="status" aria-live="polite"></p></form>
<small>授权采用 OAuth 2.1 + PKCE；密码不会交给官端。</small></main>
<script nonce="{script_nonce}">document.getElementById('oauth-form').addEventListener('submit',function(){{
var b=document.getElementById('connect-button'),s=document.getElementById('submit-status');
b.disabled=true;b.textContent='正在连接…';s.textContent='正在验证并返回 Claude，请稍候。';
setTimeout(function(){{b.disabled=false;b.textContent='重新确认连接';s.textContent='如果仍未跳转，请按回车重试一次。';}},15000);
}});</script></body></html>"""
        return HTMLResponse(body, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "Content-Security-Policy": f"default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-{script_nonce}'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"})

    def _login_limited(self, address: str) -> bool:
        now = time.time()
        attempts = self._failed_logins[address]
        while attempts and attempts[0] < now - 600:
            attempts.popleft()
        return len(attempts) >= 5

    async def login_callback(self, request: Request) -> Response:
        address = request.client.host if request.client else "unknown"
        logger.info("OAuth callback received")
        if self._login_limited(address):
            return HTMLResponse("<h1>尝试次数过多，请十分钟后再试。</h1>", status_code=429)
        form = await request.form()
        state = form.get("state")
        password = form.get("password")
        if not isinstance(state, str) or not isinstance(password, str):
            return HTMLResponse("<h1>连接请求不完整。</h1>", status_code=400)
        pending_key = _secret_key(state)
        self._reload_store()
        pending = self._store["pending_authorizations"].get(pending_key)
        if not pending or pending.get("expires_at", 0) < time.time():
            logger.warning("OAuth callback state missing or expired (pending=%d)", len(self._store["pending_authorizations"]))
            return HTMLResponse("<h1>这次连接请求已失效，请回到官端重试。</h1>", status_code=400)
        logger.info("OAuth callback state accepted for client %.12s", str(pending.get("client_id") or ""))
        if not self.verify_owner_password(password):
            self._failed_logins[address].append(time.time())
            async with self._lock:
                self._reload_store()
                self._audit("login_failed", client_id=pending.get("client_id", ""), address_hash=_secret_key(address)[:12])
                self._write_store()
            return HTMLResponse("<h1>密码不对，请返回重试。</h1>", status_code=401)

        raw_code = secrets.token_urlsafe(32)
        code = AuthorizationCode(
            code=raw_code,
            client_id=str(pending["client_id"]),
            redirect_uri=pending["redirect_uri"],
            redirect_uri_provided_explicitly=bool(pending["redirect_uri_provided_explicitly"]),
            expires_at=time.time() + 300,
            scopes=list(pending.get("scopes") or [self.scope]),
            code_challenge=str(pending["code_challenge"]),
            resource=str(pending.get("resource") or self.resource_url),
            subject=self.owner_name,
        )
        async with self._lock:
            self._reload_store()
            durable_pending = self._store["pending_authorizations"].get(pending_key)
            if not durable_pending or durable_pending.get("expires_at", 0) < time.time():
                logger.warning("OAuth callback state disappeared before commit")
                return HTMLResponse("<h1>这次连接请求已失效，请回到官端重试。</h1>", status_code=400)
            self._store["authorization_codes"][_secret_key(raw_code)] = code.model_dump(mode="json", exclude={"code"})
            # Keep the password-page state for its short TTL. Browsers and
            # official clients sometimes submit the form twice; issuing a new
            # one-time code is safer than showing a false "expired" page.
            durable_pending["successful_callbacks"] = int(durable_pending.get("successful_callbacks", 0)) + 1
            durable_pending["last_callback_at"] = int(time.time())
            self._audit("owner_authorized", client_id=code.client_id)
            self._write_store()
        logger.info("OAuth owner authorization completed for client %.12s", code.client_id)
        redirect = construct_redirect_uri(str(code.redirect_uri), code=raw_code, state=pending.get("oauth_state"))
        return RedirectResponse(redirect, status_code=302)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        self._reload_store()
        record = self._store["authorization_codes"].get(_secret_key(authorization_code))
        if not record or record.get("client_id") != client.client_id:
            return None
        return AuthorizationCode.model_validate({**record, "code": authorization_code})

    def _new_token_pair(self, *, client_id: str, scopes: list[str], resource: str | None, subject: str | None) -> OAuthToken:
        now = int(time.time())
        raw_access = secrets.token_urlsafe(48)
        raw_refresh = secrets.token_urlsafe(48)
        access = AccessToken(
            token=raw_access,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + 3600,
            resource=resource or self.resource_url,
            subject=subject or self.owner_name,
            claims={"iss": self.issuer_url},
        )
        refresh = OmbreRefreshToken(
            token=raw_refresh,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + 30 * 86400,
            subject=subject or self.owner_name,
            resource=resource or self.resource_url,
        )
        self._store["access_tokens"][_secret_key(raw_access)] = access.model_dump(mode="json", exclude={"token"})
        self._store["refresh_tokens"][_secret_key(raw_refresh)] = refresh.model_dump(mode="json", exclude={"token"})
        return OAuthToken(
            access_token=raw_access,
            refresh_token=raw_refresh,
            token_type="Bearer",
            expires_in=3600,
            scope=" ".join(scopes),
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        key = _secret_key(authorization_code.code)
        async with self._lock:
            self._reload_store()
            if key not in self._store["authorization_codes"]:
                raise TokenError(error="invalid_grant", error_description="Authorization code was already used")
            self._store["authorization_codes"].pop(key, None)
            tokens = self._new_token_pair(
                client_id=str(client.client_id),
                scopes=authorization_code.scopes,
                resource=authorization_code.resource,
                subject=authorization_code.subject,
            )
            self._audit("token_issued", client_id=client.client_id or "")
            self._write_store()
        return tokens

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> OmbreRefreshToken | None:
        self._reload_store()
        record = self._store["refresh_tokens"].get(_secret_key(refresh_token))
        if not record or record.get("client_id") != client.client_id:
            return None
        return OmbreRefreshToken.model_validate({**record, "token": refresh_token})

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: OmbreRefreshToken, scopes: list[str]
    ) -> OAuthToken:
        if scopes and not set(scopes).issubset(set(refresh_token.scopes)):
            raise TokenError(error="invalid_scope", error_description="Requested scope was not originally granted")
        key = _secret_key(refresh_token.token)
        async with self._lock:
            self._reload_store()
            if key not in self._store["refresh_tokens"]:
                raise TokenError(error="invalid_grant", error_description="Refresh token was already used")
            self._store["refresh_tokens"].pop(key, None)
            tokens = self._new_token_pair(
                client_id=str(client.client_id),
                scopes=scopes or refresh_token.scopes,
                resource=refresh_token.resource,
                subject=refresh_token.subject,
            )
            self._audit("token_refreshed", client_id=client.client_id or "")
            self._write_store()
        return tokens

    async def load_access_token(self, token: str) -> AccessToken | None:
        if self.service_token and hmac.compare_digest(token, self.service_token):
            return AccessToken(
                token=token,
                client_id="kelo-home-service",
                scopes=[self.scope],
                resource=self.resource_url,
                subject="kelo-home",
                claims={"iss": "ombre-service-token"},
            )
        self._reload_store()
        record = self._store["access_tokens"].get(_secret_key(token))
        if not record or (record.get("expires_at") and record["expires_at"] < time.time()):
            return None
        return AccessToken.model_validate({**record, "token": token})

    async def revoke_token(self, token: AccessToken | OmbreRefreshToken) -> None:
        raw = getattr(token, "token", "")
        if not raw or (self.service_token and hmac.compare_digest(raw, self.service_token)):
            return
        async with self._lock:
            self._reload_store()
            self._store["access_tokens"].pop(_secret_key(raw), None)
            self._store["refresh_tokens"].pop(_secret_key(raw), None)
            self._audit("token_revoked", client_id=getattr(token, "client_id", ""))
            self._write_store()


def install_oauth_login_routes(mcp: Any, provider: OmbreOAuthProvider) -> None:
    @mcp.custom_route("/oauth/login", methods=["GET"])
    async def oauth_login(request: Request) -> Response:
        return await provider.login_page(request)

    @mcp.custom_route("/oauth/callback", methods=["POST"])
    async def oauth_callback(request: Request) -> Response:
        return await provider.login_callback(request)
