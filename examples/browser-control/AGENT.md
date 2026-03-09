# AGENT.md

## 目的

本文件定义 `browser-control` 项目的 Agent 工作流入口、结构化上下文位置，以及执行任务时需要遵守的最小工程规范。

## 一、工作流元数据位置

和 Agent 流水线相关的结构化文件统一放在根目录下的 `.project/` 目录中，不再散落在项目根目录或普通文档目录下。

## 二、`.project/` 目录结构

```text
.project/
  project-context.md
  features.json
  todos.yaml
  AI_Agent_Workflow_Level2.md
  specs/
  threads/
  events/
```

## 三、各文件职责

- `.project/project-context.md`：项目背景、构建方式、约束、当前优先事项
- `.project/features.json`：Feature 状态机
- `.project/todos.yaml`：当前可执行 TODO 合同
- `.project/specs/`：每个 feature 的 spec 文档
- `.project/threads/`：任务线程记录，用于跨会话续跑
- `.project/events/`：事件归档，用于记录关键操作和验证结果
- `.project/AI_Agent_Workflow_Level2.md`：Level2 工作流说明

### 文档提炼规则

项目中的原始文档应按职责提炼进 `.project`，不要直接复制全文。

- `README.md` / 快速上手类文档：提炼到 `.project/project-context.md`
- 设计文档 / 架构文档：提炼到对应 `spec` 和 `project-context`
- 当前推进状态与下一步动作：提炼到 `thread`
- 本次提炼动作本身：记录到 `event`

原始文档继续保留为详细参考入口，但 `.project` 只保存“当前推进真正需要的稳定信息”。

### 文档归档规则

当原始文档已经完成提炼后：

- 老文档迁移到 `.project/docs/` 统一归档
- 根目录 `README` 保留为轻量入口页
- `.project` 内引用统一切到 `.project/docs/` 新路径

## 四、项目运行与实验规则

### 1. Python 依赖与命令入口统一使用 `uv`

- 默认使用 `uv sync` 安装依赖
- 默认使用 `uv run ...` 执行 Python、Modal 等命令
- 不要随意混用 `pip`、`python`、`conda`，除非当前步骤有明确例外

### 2. BrowserGym 观察与动作格式规则

- 训练和评估时，观察输入优先使用 `axtree_txt`
- 不要依赖 `text` 字段作为主要观察内容
- action 必须是 Python 函数调用字符串，如 `click('13')`
- bid 必须带引号，不要输出 `click(13)` 这种格式
- 如果环境返回 `last_action_error` 或 `error`，应优先把它视为动作格式或执行失败的诊断入口

### 3. 先检查外部前置条件，再判断代码问题

这个项目的训练和评估依赖多个外部系统。执行前优先检查：

1. `uv sync` 是否完成
2. Modal 是否已认证
3. BrowserGym / OpenEnv / Docker 服务是否就绪
4. WandB、模型路径、Volume 等外部前置条件是否可用

不要在外部条件未确认前，就直接把失败归因到代码实现。

### 4. TRL / vLLM / rollout 行为以实际运行环境为准

- `rollout_func` 与 `vllm_mode` 的行为可能随 TRL 版本变化
- 不要只依赖宿主机源码推断容器或远端环境中的真实行为
- 涉及 rollout、TRL、vLLM 时，优先记录并核对实际运行环境版本和实测结果

### 5. 基线 feature 不要混入重型扩展方向

当前首个 feature 只覆盖训练/评估基线。以下方向应拆成后续独立 feature：

1. GGUF 转换
2. 检查点管理与提取
3. Android / LeapSDK 集成

不要在 `FEAT-001` 中顺手扩展到这些更重的方向。

### 6. BrowserGym 训练提示与动作生成约束

- 训练与评估提示应围绕 `goal + axtree_txt + error` 组织，不要退化成泛化自然语言描述
- 生成结果应优先被解析为单个 action，而不是解释性文本
- 如需修改 action 解析逻辑，先保证 `click('13')` / `fill('5', 'text')` 这类标准调用格式持续可用

### 7. 检查点、GGUF、Android 方向按独立主线管理

- 检查点下载、存储、压缩与提取视为 MLOps 主线
- GGUF 转换与量化视为模型构建主线
- Android / LeapSDK 相关工作视为部署集成主线

不要把这些主线的脚本、目录或验证目标混进当前训练/评估基线任务里。

## 五、项目初始化标准动作

### 0. 初始化模板以全局目录为唯一标准源

- `~/.config/opencode/.project/templates/`

不要在单个项目里长期维护 `.project/templates/` 本地模板副本，除非该项目确实需要特化模板。

## 六、开始任务前的默认读取顺序

1. `.project/project-context.md`
2. `.project/features.json`
3. `.project/todos.yaml`
4. `.project/specs/browser-control-training-baseline.md`
5. `.project/threads/THREAD-browser-control-training-baseline.md`

## 七、当前首个业务 Feature

- `FEAT-001`: `browser-control-training-baseline`

当前建议优先执行：

- `TODO-001`: 盘点并验证 `uv sync` 与 debug 训练前置条件

## 归档文档

- 历史 README 归档：`.project/docs/browser-control-readme-archive.md`
- 历史开发指南归档：`.project/docs/browser-control-agents-guide.md`
- 历史技能索引归档：`.project/docs/browser-control-skill-index.md`
- 历史 learned 经验归档：`.project/docs/skills-learned/`

## 八、提交规则

- 提交代码时，不要提交项目中的 `.project/` 和 `.learnings/` 目录文件，除非用户明确要求把这些工作流/学习记录一并纳入版本控制
