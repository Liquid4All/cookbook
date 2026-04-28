# browser-control

使用 `LFM2-350M + GRPO + BrowserGym/OpenEnv` 训练浏览器控制模型，并逐步扩展到本地 Docker 训练、GGUF 转换和 Android / LeapSDK 部署方向。

## What It Includes

- `src/browser_control/`：训练、评估、配置与 Modal 基础设施
- `configs/`：debug / full / LoRA / local 配置
- `docker/`：本地 BrowserGym 与训练容器主线
- `scripts/`：GGUF 转换与辅助脚本
- `android/`：后续部署集成方向

## Quick Start

```bash
# 依赖同步
uv sync

# debug 训练（更稳的入口）
uv run python -m modal run -m src.browser_control.fine_tune --config-file-name lfm2_350m_debug.yaml

# 评估
make evaluation
```

## Project Entry Points

- 项目上下文：`.project/project-context.md`
- 当前 feature spec：`.project/specs/browser-control-training-baseline.md`
- 当前任务线程：`.project/threads/THREAD-browser-control-training-baseline.md`
- Agent 工作流入口：`AGENT.md`
- 技能入口：`.project/SKILLS.md`

## Archived Docs

- 历史 README 归档：`.project/docs/browser-control-readme-archive.md`
- 开发与专题归档：`.project/docs/`

## Notes

- 当前首个主线是训练/评估基线：`FEAT-001`
- 后续主线已拆分为 Docker 本地训练、GGUF/检查点、Android/LeapSDK 三个独立 feature 草案
- 长期有效信息已经提炼进 `.project/`
