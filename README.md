# AI Coding Agent (Experimental Project) 

一个实验性质的 AI coding agent：以命令行应用（CLI/TUI）形式运行，围绕“计划-执行-反馈”的 agentic loop 组织能力，并通过工具系统与外部世界交互。

This is a experimental AI coding agent that runs as a CLI/TUI application. It follows a plan-execute-feedback loop and uses tools to interact with your project and external resources.

## Capabilities / 能力概览

- Read & understand codebase / 读取并理解代码库
- Write & edit files / 写入与编辑文件
- Run shell commands / 执行命令行指令
- Search project (glob/grep) / 项目内搜索（glob/grep）
- Web search & fetch / 联网搜索与网页抓取
- Streaming responses / 流式输出（边生成边显示）
- Retry with exponential backoff / 请求失败重试（指数退避）

## Agent Architecture / 架构模块

- CLI application / 命令行应用
  - Single-run mode & interactive mode / 单次运行与交互模式
  - Terminal UI (TUI) with rich output / 终端 UI（更好的输出展示）

- Agentic loop / 智能体循环
  - Plans, selects tools, executes, and continues until done / 规划-选工具-执行-迭代直到完成
  - Loop detection / 循环检测（发现卡住时自我纠正）

- Tools / 工具系统
  - Tool base abstraction + tool registry / 工具基类 + 工具注册
  - File tools: read/write/edit / 文件工具：读/写/编辑
  - Shell tool / 命令执行工具
  - List directory + glob + grep / 列目录 + glob + grep
  - Web tools: search + fetch / 网络工具：搜索 + 抓取
  - Todos / planning tool / 待办与规划工具
  - Memory tool / 记忆工具（跨会话保留关键信息）

- Context management / 上下文管理
  - Compaction / 压缩：在上下文变长时总结合并
  - Pruning / 裁剪：移除不重要细节，保留关键事实
  - Sessions & checkpoints / 会话与检查点：保存/恢复执行进度

- Autonomy & safety / 自主度与安全
  - Approval system / 审批系统：对高风险操作进行确认
  - Hooks / 钩子：在关键阶段注入自定义逻辑（例如 pre/post tool call）

- Extensibility / 可扩展性
  - Sub-agents / 子代理：复杂任务拆分并行处理
  - Tool discovery / 工具发现：支持添加自定义工具
  - MCP integration / MCP（Model Context Protocol）对接第三方服务

## Typical Workflow / 典型使用方式

1) 用户给出目标（例如“修复一个 bug / 实现一个功能”）
2) Agent 读取相关文件并检索代码库
3) 生成计划（todos）并逐步执行：编辑文件、运行命令、查看输出
4) 必要时联网检索资料并抓取内容
5) 当上下文过长时自动压缩/裁剪，持续推进直到任务完成


1) User provide a goal (e.g. fix a bug / build a feature)
2) The agent reads relevant files and searches the repo
3) It plans (todos) and executes step-by-step: edits files, runs commands, inspects output
4) It can search/fetch from the web when needed
5) When context grows too long, it compacts/prunes and continues until done

## Notes / 说明

- This repository is intended for learning and experimentation, not production use.
- 本仓库以学习与实验为主，功能与安全边界需要你根据实际场景自行加强。

Inspired by: https://www.youtube.com/watch?v=3GjE_YAs03s
