.PHONY: install dev-up dev-down dev-migrate dev-server test

MODEL_ENGINE_DIR := $(abspath .)
DB_URL := postgresql://postgres:password@localhost:5432/llm_engine

# Local dev environment variables
LOCAL_ENV := \
	LOCAL=true \
	GIT_TAG=local \
	ML_INFRA_DATABASE_URL=$(DB_URL) \
	DEPLOY_SERVICE_CONFIG_PATH=$(MODEL_ENGINE_DIR)/service_configs/service_config_local.yaml

install:
	pip install -r requirements.txt -r requirements-test.txt -r requirements_override.txt
	pip install -e .

dev-up:
	docker compose -f docker-compose.local.yml up -d
	@echo "Waiting for services to be healthy..."
	@until docker compose -f docker-compose.local.yml exec postgres pg_isready -U postgres -q; do sleep 1; done
	@echo "Postgres ready."
	@until docker compose -f docker-compose.local.yml exec redis redis-cli ping | grep -q PONG; do sleep 1; done
	@echo "Redis ready."

dev-down:
	docker compose -f docker-compose.local.yml down

dev-migrate:
	$(LOCAL_ENV) bash model_engine_server/db/migrations/run_database_migration.sh

dev-server:
	$(LOCAL_ENV) start-fastapi-server --port 5000 --num-workers 1 --debug

test:
	pytest tests/unit/
