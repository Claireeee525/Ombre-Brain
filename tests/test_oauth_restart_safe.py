import base64
import hashlib
from urllib.parse import parse_qs, urlparse

from pydantic import AnyHttpUrl
from starlette.testclient import TestClient

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.fastmcp import FastMCP

from oauth_provider import OmbreOAuthProvider, install_oauth_login_routes


BASE_URL = "http://localhost"
RESOURCE_URL = f"{BASE_URL}/mcp"
SCOPE = "ombre:memory"


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def _build_app(store_path, *, service_token="home-secret"):
    provider = OmbreOAuthProvider(
        store_path=str(store_path),
        issuer_url=BASE_URL,
        resource_url=RESOURCE_URL,
        verify_owner_password=lambda password: password == "owner-password",
        service_token=service_token,
    )
    server = FastMCP(
        "OAuth test",
        host="0.0.0.0",
        stateless_http=True,
        json_response=True,
        auth_server_provider=provider,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(BASE_URL),
            resource_server_url=AnyHttpUrl(RESOURCE_URL),
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=[SCOPE],
                default_scopes=[SCOPE],
            ),
            revocation_options=RevocationOptions(enabled=True),
            required_scopes=[SCOPE],
        ),
    )
    install_oauth_login_routes(server, provider)

    @server.tool()
    async def ping() -> str:
        return "pong"

    return server.streamable_http_app()


def _initialize(client, token=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "oauth-test", "version": "1"},
            },
        },
    )


def test_oauth_flow_tokens_and_client_survive_server_restart(tmp_path):
    store_path = tmp_path / "oauth.json"
    verifier = "v" * 64
    challenge = _pkce_challenge(verifier)

    with TestClient(_build_app(store_path), base_url=BASE_URL) as client:
        unauthorized = _initialize(client)
        assert unauthorized.status_code == 401
        assert "resource_metadata=" in unauthorized.headers["www-authenticate"]

        protected = client.get("/.well-known/oauth-protected-resource/mcp")
        assert protected.status_code == 200
        assert protected.json()["resource"] == RESOURCE_URL
        assert protected.json()["authorization_servers"] == [f"{BASE_URL}/"]

        metadata = client.get("/.well-known/oauth-authorization-server")
        assert metadata.status_code == 200
        assert metadata.json()["registration_endpoint"] == f"{BASE_URL}/register"

        registered = client.post(
            "/register",
            json={
                "redirect_uris": ["https://claude.example/callback"],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scope": SCOPE,
                "client_name": "Claude official connector",
            },
        )
        assert registered.status_code == 201
        client_id = registered.json()["client_id"]

        authorized = client.get(
            "/authorize",
            params={
                "client_id": client_id,
                "redirect_uri": "https://claude.example/callback",
                "response_type": "code",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "state-123",
                "scope": SCOPE,
                "resource": RESOURCE_URL,
            },
            follow_redirects=False,
        )
        assert authorized.status_code == 302
        login_url = authorized.headers["location"]
        assert login_url.startswith(f"{BASE_URL}/oauth/login?")
        login_state = parse_qs(urlparse(login_url).query)["state"][0]
        assert login_state != "state-123"

        login_page = client.get(login_url)
        assert login_page.status_code == 200
        assert "Claude official connector" in login_page.text
        assert "owner-password" not in login_page.text

        callback = client.post(
            "/oauth/callback",
            data={"state": login_state, "password": "owner-password"},
            follow_redirects=False,
        )
        assert callback.status_code == 302
        callback_query = parse_qs(urlparse(callback.headers["location"]).query)
        code = callback_query["code"][0]
        assert callback_query["state"] == ["state-123"]

        token_response = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "https://claude.example/callback",
                "client_id": client_id,
                "code_verifier": verifier,
                "resource": RESOURCE_URL,
            },
        )
        assert token_response.status_code == 200
        tokens = token_response.json()
        assert tokens["scope"] == SCOPE
        assert _initialize(client, tokens["access_token"]).status_code == 200

    # A fresh app/provider reads the same volume-backed store.  This is the
    # acceptance that prevents a deployment restart from becoming "day two".
    with TestClient(_build_app(store_path), base_url=BASE_URL) as restarted:
        assert _initialize(restarted, tokens["access_token"]).status_code == 200
        refreshed = restarted.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": client_id,
                "resource": RESOURCE_URL,
            },
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["refresh_token"] != tokens["refresh_token"]
        assert _initialize(restarted, refreshed.json()["access_token"]).status_code == 200
        revoked = restarted.post(
            "/revoke",
            data={
                "token": refreshed.json()["access_token"],
                "token_type_hint": "access_token",
                "client_id": client_id,
                "client_secret": "",
            },
        )
        assert revoked.status_code == 200
        assert _initialize(restarted, refreshed.json()["access_token"]).status_code == 401
        assert store_path.stat().st_mode & 0o077 == 0


def test_private_home_service_token_still_works_with_oauth(tmp_path):
    with TestClient(
        _build_app(tmp_path / "oauth.json", service_token="home-secret"), base_url=BASE_URL
    ) as client:
        assert _initialize(client, "home-secret").status_code == 200
        assert _initialize(client, "wrong-secret").status_code == 401
