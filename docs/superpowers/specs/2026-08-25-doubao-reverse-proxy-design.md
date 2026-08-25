# 豆包反代项目设计文档

## 1. 背景与目标

在 `E:\豆包反代` 工作区中，基于开源项目 `doubao2api` 扩展一个豆包反向代理服务，目标：

- 复用 `doubao2api` 已有的 OpenAI 兼容接口、QR 扫码登录、Admin Dashboard、视频/图片/音乐生成能力。
- 新增**免费账号池**：支持多个豆包免费账号 session 文件，轮询/最少使用/故障转移，提高视频生成额度上限。
- 新增**官方火山引擎通道**：同时支持官方 API Key，与免费账号池统一路由。
- 扩展 Admin Dashboard：增加账号管理和通道配置。
- 本地 Python 直接运行，保持向后兼容：无账号配置时仍使用原有单账号 BrowserClient + QR 登录模式。

## 2. 总体架构

```
Client / OpenAI SDK / Web Admin
        │
        ▼
FastAPI（现有 unified_server.py）
        │
        ├── ChannelBackend（通道抽象层）
        │       ├── FreeAccountBackend（免费账号池）
        │       │     ├── AccountPool（多 session 文件 → 多个 DoubaoChatClient）
        │       │     └── 轮询/最少使用/故障转移
        │       └── VolcanoBackend（官方火山引擎 API）
        │
        └── Admin 扩展：账号管理 + 通道状态
```

- 保留现有 HTTP 端点、SSE 流式、鉴权、日志、Admin 页面。
- 不重写 `BrowserClient`；它作为免费账号池的**可选后端**，以及无账号配置时的回退模式。

## 3. 模块划分

### 3.1 `channel.py`：通道抽象层

定义统一后端接口，方法签名与现有 `BrowserClient` 保持一致，降低对 `unified_server.py` 的改动：

- `is_ready() -> bool`
- `chat(...)`
- `chat_completion(...)`
- `chat_stream_completion(...)`
- `generate_image(...)`
- `generate_video(...)`
- `generate_music(...)`
- `upload_file(...)`
- `upload_image(...)`
- `chat_with_file(...)`

实现两个后端：

- `FreeAccountBackend`：内部持有 `AccountPool`，把请求转发给选中的账号客户端。
- `VolcanoBackend`：调用火山引擎官方 API，内部处理异步任务（视频）并返回统一结果。

### 3.2 `account_pool.py`：免费账号池

- **配置**：环境变量 `DOUBAO_ACCOUNTS_DIR`（默认 `./accounts`），目录下每个 `*.json` 为豆包 session 文件（格式兼容 `.doubao_session.json`）。
- 可选 `accounts.json`（根目录或账号目录）描述账号：`name`、`session_file`、`enabled`、`weight`、`remark`、`backend`（`http` 或 `browser`，默认 `http`）。
- 每个账号封装为一个 `AccountEntry`：
  - `name`
  - `session_file`
  - `enabled`
  - `weight`
  - `client`（`DoubaoChatClient` 或 `BrowserClient`）
  - 统计：`success_count`、`failure_count`、`last_used_at`、`last_error_code`、`needs_captcha`、`consecutive_failures`
  - 每日额度计数：`daily_quota_used`、`quota_date`（内存态，可选持久化）
- **选择策略**：默认 round-robin；可配置 `least_used`。自动跳过 `disabled`、`needs_captcha`、连续失败超过阈值的账号。
- **生命周期**：实现 `AsyncContextManager`；在 FastAPI `lifespan` 中 `start()` 所有账号客户端，退出时 `stop()`。
- **故障转移**：
  - 对同步操作（chat、image、music）可自动重试下一个健康账号。
  - 对视频生成（异步任务）只重试连接/认证阶段错误，不重试已创建任务后的错误，避免重复消耗额度。
- **后向兼容**：
  - `accounts/` 不存在且未配置任何账号时，启动原有单账号 `BrowserClient` 模式。
  - 根目录存在 `.doubao_session.json` 时，自动导入为 `default` 账号。
  - 设置 `DOUBAO_COOKIE` 环境变量时，自动创建一个账号。

### 3.3 `volcano.py`：官方火山引擎通道

- **环境变量**：
  - `VOLC_API_KEY`
  - `VOLC_BASE_URL`（默认 `https://ark.cn-beijing.volces.com/api/v3`）
  - `VOLC_CHAT_MODEL`
  - `VOLC_IMAGE_MODEL`
  - `VOLC_VIDEO_MODEL`
- **能力**：
  - 聊天：OpenAI 兼容 `/chat/completions`
  - 图片：`/images/generations`
  - 视频：官方异步任务 API（创建任务 → 轮询 → 返回视频 URL），对调用方透明
- **路由模型**：
  - `volc-chat`
  - `volc-image`
  - `volc-video`
- 若免费池不可用且已配置官方 Key，原 `doubao-*` 模型可回退到官方通道。

### 3.4 `unified_server.py` 改造

- 将 `_get_client()` 改为 `_get_backend()`，返回当前请求应使用的后端。
- 根据模型名选择后端：
  - `doubao-*` → 免费账号池（或回退官方/单账号 BrowserClient）
  - `volc-*` → 官方火山引擎
- `/v1/models` 动态列出可用模型。
- `/health` 返回池子统计和官方通道状态。
- 保留现有鉴权、限流、请求日志、SSE。

### 3.5 Admin 扩展

在现有 Vue 3 面板增加两个 Tab：

1. **账号管理**
   - 列表：名称、状态、启用/停用、成功/失败、最后使用、风控标记、每日额度
   - 操作：添加 session 文件（multipart）、删除、停用/启用、单个探活
2. **通道配置**
   - 显示官方火山引擎是否已配置、当前默认通道
   - 测试官方 API 连通性

新增 API：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/admin/api/accounts` | GET | 列出所有账号状态 |
| `/admin/api/accounts` | POST | 添加 session 文件（multipart） |
| `/admin/api/accounts/{name}` | DELETE | 移除账号 |
| `/admin/api/accounts/{name}/toggle` | POST | 启用/停用 |
| `/admin/api/accounts/{name}/probe` | POST | 探活 |
| `/admin/api/channel` | GET | 返回通道配置状态 |
| `/admin/api/channel/volc` | POST | 配置火山引擎（保存 API Key） |

## 4. 数据流

1. 客户端请求到达 FastAPI。
2. `_get_backend()` 根据模型名选择后端。
3. 免费池后端从 `AccountPool` 选择一个健康账号，调用账号客户端。
4. 成功则更新账号统计；失败则按策略标记账号并按需切换。
5. 官方后端直接调用火山引擎 API，异步任务内部轮询完成后返回统一响应。
6. Admin API 直接操作 `AccountPool` 和火山配置。

## 5. 错误处理

- 账号级错误独立记录，避免一个账号风控拖垮整个服务。
- 免费池对同步操作可自动重试下一个健康账号；视频生成不自动重试已创建任务后的错误。
- 全部账号不可用时返回 `503`，错误信息包含可用账号数。
- 官方通道未配置时，`volc-*` 模型返回 `503` 并提示配置 `VOLC_API_KEY`。
- 保留现有 `710022002` / `710022004` 风控码标记逻辑。

## 6. 配置示例

```json
{
  "accounts_dir": "./accounts",
  "accounts": [
    { "name": "主号", "session_file": "./accounts/main.json", "enabled": true, "weight": 2, "backend": "http" },
    { "name": "备号", "session_file": "./accounts/backup.json", "enabled": true, "weight": 1, "backend": "http" }
  ],
  "volcano": {
    "api_key": "",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "chat_model": "doubao-pro-32k-250715",
    "image_model": "doubao-seedream-5-0-pro-260128",
    "video_model": "doubao-seedance-1-5-pro-251215"
  }
}
```

## 7. 测试

- 单元测试：`AccountPool` 加载、轮询选择、禁用/故障跳过、故障转移。
- 单元测试：`VolcanoBackend` 请求构造（mock httpx）。
- 集成测试：用假 backend 启动 FastAPI，验证 `/v1/video/generations` 路由到正确通道。
- 保持现有 `tests/` 目录结构；现有测试依赖外部服务无法本地运行，不作为阻塞项。

## 8. 非目标

- 不实现付费会员账号管理。
- 不实现公网部署、HTTPS、复杂用户系统。
- 不重写 `doubao2api` 的底层逆向协议。
- 不保证绕过豆包官方风控或水印限制。
