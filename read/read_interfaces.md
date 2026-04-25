## `app/interfaces/` 是什么

这里放的是“接口层”，目的是 **让上层代码不绑定具体实现**。

你可以把它理解成：我们先把插口标准定好，后面要换 OpenAI / 换 MCP server / 换记忆库，都只需要换实现，不要全项目大改。

---

## 目前有哪些接口

- `contracts.py`
  - 节点/agent 的基础契约（run/execute 的形状）

- `llm.py`
  - `LLMClient`：LLM 调用接口（现在默认是 mock，方便本地跑）

- `mcp.py`
  - `HttpMCPClient`：真实 HTTP 调用（失败会 fallback 到 mock）
  - `MockMCPClient`：本地占位/降级用

- `memory.py`
  - `MemoryStore`：记忆库接口
  - `MockMemoryStore`：本地占位/降级用

---

## 一个小约定（很重要）

接口层尽量做到两点：
- **签名稳定**（上层不用跟着改）
- **可降级**（外部服务挂了也能跑流程/跑测试）