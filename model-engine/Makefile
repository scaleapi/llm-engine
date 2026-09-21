.PHONY: install dev-up dev-down dev-migrate dev-server test \
        kind-up kind-down kind-image \
        dev-service-builder dev-k8s-cacher dev-server-full

MODEL_ENGINE_DIR := $(abspath .)
DB_URL           := postgresql://postgres:password@localhost:5432/llm_engine
KIND_CLUSTER     := llm-engine
KUBE_CONTEXT     := kind-$(KIND_CLUSTER)

# ── Control-plane-only (no k8s, fake queue/docker) ──────────────────────────
LOCAL_ENV := \
	LOCAL=true \
	GIT_TAG=local \
	ML_INFRA_DATABASE_URL=$(DB_URL) \
	DEPLOY_SERVICE_CONFIG_PATH=$(MODEL_ENGINE_DIR)/service_configs/service_config_local.yaml \
	ML_INFRA_SERVICES_CONFIG_PATH=$(MODEL_ENGINE_DIR)/model_engine_server/core/configs/default.yaml

# ── Full end-to-end (real k8s via kind, real Redis queue, fake docker) ───────
FULL_LOCAL_ENV := \
	LOCAL=true \
	GIT_TAG=local \
	ML_INFRA_DATABASE_URL=$(DB_URL) \
	DEPLOY_SERVICE_CONFIG_PATH=$(MODEL_ENGINE_DIR)/service_configs/service_config_local.yaml \
	ML_INFRA_SERVICES_CONFIG_PATH=$(MODEL_ENGINE_DIR)/model_engine_server/core/configs/local-full.yaml \
	REDIS_HOST=localhost \
	REDIS_PORT=6379

# ── One-time setup ───────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt -r requirements-test.txt -r requirements_override.txt
	pip install -e .

# ── Backing services (Postgres + Redis) ─────────────────────────────────────
dev-up:
	docker compose -f docker-compose.local.yml up -d --wait

dev-down:
	docker compose -f docker-compose.local.yml down

dev-migrate:
	$(LOCAL_ENV) bash model_engine_server/db/migrations/run_database_migration.sh

# ── Control-plane-only server ────────────────────────────────────────────────
dev-server:
	$(LOCAL_ENV) start-fastapi-server --port 5000 --num-workers 1 --debug

# ── kind cluster (for full end-to-end flow) ──────────────────────────────────
kind-up:
	kind create cluster --name $(KIND_CLUSTER)
	kubectl --context $(KUBE_CONTEXT) create namespace model-engine --dry-run=client -o yaml | kubectl --context $(KUBE_CONTEXT) apply -f -

kind-down:
	kind delete cluster --name $(KIND_CLUSTER)

kind-image:
	docker build -t model-engine:local ..
	kind load docker-image model-engine:local --name $(KIND_CLUSTER)

# ── Full end-to-end processes (run each in a separate terminal) ───────────────
dev-server-full:
	$(FULL_LOCAL_ENV) start-fastapi-server --port 5000 --num-workers 1 --debug

dev-service-builder:
	$(FULL_LOCAL_ENV) celery -A model_engine_server.service_builder.celery worker --loglevel=info --concurrency=2

dev-k8s-cacher:
	$(FULL_LOCAL_ENV) python model_engine_server/entrypoints/k8s_cache.py --sleep-interval-seconds 5

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/
