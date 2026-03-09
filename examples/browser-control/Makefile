##########################
# Modal云端训练 (原有)
##########################

fine-tune:
	uv run modal run -m src.browser_control.fine_tune --config-file-name $(config)

evaluation:
	uv run python -m src.browser_control.evaluate

##########################
# Docker本地训练 (新增)
##########################

.PHONY: docker-build
docker-build:
	@echo "构建Docker镜像..."
	cd docker && docker compose -f docker-compose.training.yml build

.PHONY: docker-up-browsergym
docker-up-browsergym:
	@echo "启动BrowserGym服务..."
	cd docker && docker compose -f docker-compose.training.yml up -d browsergym

.PHONY: test-browsergym
test-browsergym:
	@echo "测试BrowserGym环境..."
	cd docker && bash scripts/test_browsergym.sh

.PHONY: fine-tune-local
fine-tune-local:
	@echo "启动本地Docker训练（调试配置）..."
	cd docker && CONFIG_FILE=lfm2_350m_local.yaml bash scripts/train.sh

.PHONY: fine-tune-local-full
fine-tune-local-full:
	@echo "启动本地Docker训练（完整配置）..."
	cd docker && CONFIG_FILE=lfm2_350m_local_full.yaml bash scripts/train.sh

.PHONY: fine-tune-local-full-v2
fine-tune-local-full-v2:
	@echo "启动本地Docker训练（完整配置 v2，修复 bid 幻觉问题）..."
	cd docker && CONFIG_FILE=lfm2_350m_local_full_v2.yaml bash scripts/train.sh

.PHONY: tensorboard
tensorboard:
	@echo "启动TensorBoard (http://localhost:6006)..."
	@pkill -f "tensorboard --logdir" 2>/dev/null || true
	nohup /home/tony/.local/bin/tensorboard \
		--logdir docker/logs \
		--host 0.0.0.0 \
		--port 6006 \
		> /tmp/tensorboard.log 2>&1 &
	@echo "TensorBoard 已启动，访问 http://localhost:6006"

.PHONY: checkpoint-download
checkpoint-download:
	@echo "下载检查点（使用方法: make checkpoint-download NAME=xxx）"
	@if [ -z "$(NAME)" ]; then \
		echo "错误: 请指定检查点名称，例如: make checkpoint-download NAME=LFM2-350M-browsergym-20260223-120000"; \
		exit 1; \
	fi
	cd docker && bash scripts/download_checkpoint.sh $(NAME)

.PHONY: docker-down
docker-down:
	@echo "停止并清理Docker环境..."
	cd docker && docker compose -f docker-compose.training.yml down -v

.PHONY: docker-logs
docker-logs:
	@echo "查看训练日志..."
	docker logs -f lfm2-grpo-training

.PHONY: docker-status
docker-status:
	@echo "Docker容器状态："
	docker ps | grep -E "browsergym|training|tensorboard" || echo "无运行中的容器"