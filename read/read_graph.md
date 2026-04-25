## `app/graph/` 是什么

这里是整个系统“怎么跑起来”的地方：用 LangGraph 把节点串成流程，并支持中断与恢复。

---

## 需要关注的文件

- `state.py`
  - `MacpState`：全局数据字典（团队最重要的契约）
  - 后面所有节点都只读/只写这里面的字段，不要随便加“隐形字段”

- `main_graph.py`
  - 主图编排：`TM -> human_review -> SG -> Eva -> (reflect/stop) -> memory`（这里我不太清楚整个流程是不是有问题，因为这个memory让我有些头疼，我正在想办法测试，不过你不必担心，哪怕我们的整体流程图不对，但是不影响单个功能测试）
  - `human_review` 是 human-in-the-loop 中断点
  - `checkpoint` 用 `thread_id` 做“同一次任务”的恢复键

- `checkpoint.py`
  - checkpointer 管理（默认 SQLite，也支持 InMemory 模式用于测试）

- `routes.py`
  - 条件路由：Eva 不通过就回到 SG，直到达到上限或通过

---

## 怎么理解 human-in-the-loop

流程跑到 `human_review` 会暂停，外部需要给一个 resume payload，比如：
- `approved: true/false`
- 可选：覆盖 `constraints`、`subtasks`

这保证“关键决策点”一定有人确认，避免越跑越偏。

---

## checkpoint 是干什么的

简单说：你可以中断、关掉程序、再用同一个 `thread_id` 继续跑。

项目里默认把 checkpoint 存到 `app_data/checkpoints/` 下面的 sqlite 文件。