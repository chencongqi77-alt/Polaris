## `app/agents/` 是什么

这里放的是“Agent 的通用底座”，业务的 TM / SG / Eva 后面都会继承它。

目前我们不追求 TM/SG/Eva 的智能程度，只关心：**agent 这套壳子是不是真能跑**。

这部分陈从启来负责，如果你没有时间细致的搞，就先不管了。
（但是你应该也是要看的，因为这个你要测base能不能正常用mcp、qdrant这些东西）

---

## 需要关注的文件

- `base.py`
  - `BaseAgent`：统一能力入口（LLM 调用 / MCP 工具调用 / prompt 管理 / 重试 / 状态与上下文）
  - `execute()`：带重试和错误记录的统一执行入口

- `io_models.py`
  - `AgentInput` / `AgentOutput`：统一输入输出封装（方便后面做日志、trace、协议一致性）

- `tm.py` / `sg.py` / `eva.py`
  - 现在只是“可运行示例”，不要在这阶段花太多时间优化逻辑
  - 他们最后会继承base的能力，然后他们三个根据自己的需求再加

---

## BaseAgent 现在提供了什么（框架角度）

- **LLM 调用**：`call_llm()`
- **MCP 工具调用**：`call_mcp_tool()`
- **prompt 模板**：`PromptRegistry` + `render_prompt()`
- **重试与错误收集**：`execute()` 会把失败记录进 `state.metadata["errors"]`
- **共享上下文**：`build_context()`
- **agent 私有状态**：`get_agent_state()`（写在 `state.metadata["agent_state"][agent_name]`）
- **共享记忆入口**：`retrieve_memories()`（写在 `state.metadata["memory_hits"][agent_name]`）
