# 环境变量参考

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `OMBRE_API_KEY` | 是 | — | Gemini / OpenAI-compatible API Key，用于脱水(dehydration)和向量嵌入 |
| `OMBRE_BASE_URL` | 否 | `https://generativelanguage.googleapis.com/v1beta/openai/` | API Base URL（可替换为代理或兼容接口） |
| `OMBRE_TRANSPORT` | 否 | `stdio` | MCP 传输模式：`stdio` / `sse` / `streamable-http` |
| `OMBRE_PORT` | 否 | `8000` | HTTP/SSE 模式监听端口（仅 `sse` / `streamable-http` 生效） |
| `OMBRE_BUCKETS_DIR` | 否 | `./buckets` | 记忆桶文件存放目录（绑定 Docker Volume 时务必设置） |
| `OMBRE_HOOK_URL` | 否 | — | Breath/Dream Webhook 推送地址（POST JSON），留空则不推送 |
| `OMBRE_HOOK_SKIP` | 否 | `false` | 设为 `true`/`1`/`yes` 跳过 Webhook 推送（即使 `OMBRE_HOOK_URL` 已设置） |
| `OMBRE_DASHBOARD_PASSWORD` | 否 | — | 预设 Dashboard 访问密码；设置后覆盖文件存储的密码，首次访问不弹设置向导 |
| `OMBRE_HOME_READ_TOKEN` | 否 | — | 小家服务读取 `/api/somatic/summary` 的专用 Bearer token；只用于脱敏身体摘要，不要复用模型 API Key 或 Dashboard 密码 |
| `OMBRE_MCP_TOKEN` | 否 | `OMBRE_HOME_READ_TOKEN` | 远程 MCP 的 Bearer token；留空时复用小家读取 token，避免维护两份密钥 |
| `OMBRE_OAUTH_ENABLED` | 否 | `false` | 启用 MCP 标准 OAuth 2.1；生产公网应设为 `true`，启用前必须已有 Dashboard 密码 |
| `OMBRE_PUBLIC_URL` | 否 | `https://kelo-brain.zeabur.app` | OAuth 发行者与 MCP 资源的公网 HTTPS 根地址，不带末尾 `/` |
| `OMBRE_MCP_REQUIRE_AUTH` | 否 | `false` | 旧式静态 Bearer 门禁，仅作兼容回退；OAuth 开启时忽略 |
| `OMBRE_DEHYDRATION_MODEL` | 否 | `deepseek-chat` | 脱水/打标/合并/拆分用的 LLM 模型名（覆盖 `dehydration.model`） |
| `OMBRE_DEHYDRATION_BASE_URL` | 否 | `https://api.deepseek.com/v1` | 脱水模型的 API Base URL（覆盖 `dehydration.base_url`） |
| `OMBRE_MODEL` | 否 | — | `OMBRE_DEHYDRATION_MODEL` 的别名（前者优先） |
| `OMBRE_EMBEDDING_MODEL` | 否 | `gemini-embedding-001` | 向量嵌入模型名（覆盖 `embedding.model`） |
| `OMBRE_EMBEDDING_BASE_URL` | 否 | — | 向量嵌入的 API Base URL（覆盖 `embedding.base_url`；留空则复用脱水配置） |

## 说明

- `OMBRE_API_KEY` 也可在 `config.yaml` 的 `dehydration.api_key` / `embedding.api_key` 中设置，但**强烈建议**通过环境变量传入，避免密钥写入文件。
- `OMBRE_DASHBOARD_PASSWORD` 设置后，Dashboard 的"修改密码"功能将被禁用（显示提示，建议直接修改环境变量）。未设置则密码存储在 `{buckets_dir}/.dashboard_auth.json`（SHA-256 + salt）。
- 若启用小家里的「珂洛此刻」，请在 Ombre 与小家两个 Zeabur 服务中设置同一个高强度随机 `OMBRE_HOME_READ_TOKEN`。token 只放环境变量和 `Authorization: Bearer` 请求头，不放 URL、仓库或日志。
- 小家会在服务端请求头中携带 `OMBRE_MCP_TOKEN`，浏览器端看不到。开启 OAuth 后，同一个 token 作为小家专用服务凭据继续有效；官端走动态注册、PKCE、访问令牌和刷新令牌。
- OAuth 的客户端注册、授权码、访问令牌和刷新令牌写在 `{buckets_dir}/.oauth_state.json`，文件权限为 `0600`，因此容器重启不会丢失官端登录。访问令牌 1 小时、刷新令牌 30 天，并在刷新时轮换。
- 登录确认页保留 30 分钟；回调会重新读取持久卷并容忍浏览器重复提交，避免请求落到不同 worker 或按钮重复提交时误报“连接请求已失效”。
- 官端授权页复用 Dashboard 密码校验；密码不会写进 OAuth 状态文件，也不会交给官端。连续输错 5 次会暂缓 10 分钟。

## Webhook 推送格式 (`OMBRE_HOOK_URL`)

设置 `OMBRE_HOOK_URL` 后，Ombre Brain 会在以下事件发生时**异步**（fire-and-forget，5 秒超时）`POST` JSON 到该 URL：

| 事件名 (`event`) | 触发时机 | `payload` 字段 |
|------------------|----------|----------------|
| `breath` | MCP 工具 `breath()` 返回时 | `mode` (`ok`/`empty`), `matches`, `chars` |
| `dream` | MCP 工具 `dream()` 返回时 | `recent`, `chars` |
| `breath_hook` | HTTP `GET /breath-hook` 命中（SessionStart 钩子） | `surfaced`, `chars` |
| `dream_hook` | HTTP `GET /dream-hook` 命中 | `surfaced`, `chars` |

请求体结构（JSON）：

```json
{
  "event": "breath",
  "timestamp": 1730000000.123,
  "payload": { "...": "..." }
}
```

Webhook 推送失败仅在服务日志中以 WARNING 级别记录，**不会影响 MCP 工具的正常返回**。
