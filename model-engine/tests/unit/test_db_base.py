import sys
from unittest.mock import MagicMock, patch


def _remove_pytest_from_modules():
    """
    Pop 'pytest' from sys.modules so that get_engine_url falls through
    to the real cloud-provider logic instead of the in-test PostgreSQL branch.
    Returns the saved module so the caller can restore it.
    """
    return sys.modules.pop("pytest", None)


def _restore_pytest_module(saved):
    if saved is not None:
        sys.modules["pytest"] = saved


class TestGetEngineUrlGcp:
    """Tests for the GCP branch of get_engine_url."""

    def test_gcp_with_secret_manager_read_write(self, monkeypatch):
        """GCP + DB_SECRET_NAME set → calls GCP Secret Manager, uses clusterHost."""
        monkeypatch.delenv("ML_INFRA_DATABASE_URL", raising=False)
        monkeypatch.setenv("DB_SECRET_NAME", "my-db-secret")
        monkeypatch.setenv("DB_SECRET_GCP_PROJECT_ID", "my-gcp-project")

        mock_creds = {
            "username": "dbuser",
            "password": "dbpass",
            "clusterHost": "db.internal",
            "clusterHostRo": "dbro.internal",
            "port": 5432,
            "dbname": "llm_engine",
        }
        mock_get_key_file = MagicMock(return_value=mock_creds)
        mock_secrets_module = MagicMock()
        mock_secrets_module.get_key_file = mock_get_key_file

        saved = _remove_pytest_from_modules()
        try:
            with (
                patch("model_engine_server.db.base.infra_config") as mock_infra_config,
                patch.dict(
                    sys.modules,
                    {"model_engine_server.core.gcp.secrets": mock_secrets_module},
                ),
            ):
                mock_infra = MagicMock()
                mock_infra.cloud_provider = "gcp"
                mock_infra.env = "test"
                mock_infra_config.return_value = mock_infra

                from model_engine_server.db.base import get_engine_url

                result = get_engine_url(read_only=False)

        finally:
            _restore_pytest_module(saved)

        assert result.url == "postgresql://dbuser:dbpass@db.internal:5432/llm_engine"
        mock_get_key_file.assert_called_once_with("my-db-secret", "my-gcp-project")

    def test_gcp_with_secret_manager_read_only(self, monkeypatch):
        """GCP + DB_SECRET_NAME set + read_only=True → uses clusterHostRo."""
        monkeypatch.delenv("ML_INFRA_DATABASE_URL", raising=False)
        monkeypatch.setenv("DB_SECRET_NAME", "my-db-secret")
        monkeypatch.setenv("DB_SECRET_GCP_PROJECT_ID", "my-gcp-project")

        mock_creds = {
            "username": "dbuser",
            "password": "dbpass",
            "clusterHost": "db.internal",
            "clusterHostRo": "dbro.internal",
            "port": 5432,
            "dbname": "llm_engine",
        }
        mock_secrets_module = MagicMock()
        mock_secrets_module.get_key_file = MagicMock(return_value=mock_creds)

        saved = _remove_pytest_from_modules()
        try:
            with (
                patch("model_engine_server.db.base.infra_config") as mock_infra_config,
                patch.dict(
                    sys.modules,
                    {"model_engine_server.core.gcp.secrets": mock_secrets_module},
                ),
            ):
                mock_infra = MagicMock()
                mock_infra.cloud_provider = "gcp"
                mock_infra.env = "test"
                mock_infra_config.return_value = mock_infra

                from model_engine_server.db.base import get_engine_url

                result = get_engine_url(read_only=True)

        finally:
            _restore_pytest_module(saved)

        assert result.url == "postgresql://dbuser:dbpass@dbro.internal:5432/llm_engine"

    def test_gcp_without_secret_manager_falls_back_to_env_vars(self, monkeypatch):
        """GCP + no DB_SECRET_NAME → reads DB_* env vars directly."""
        monkeypatch.delenv("ML_INFRA_DATABASE_URL", raising=False)
        monkeypatch.delenv("DB_SECRET_NAME", raising=False)
        monkeypatch.setenv("DB_USER", "envuser")
        monkeypatch.setenv("DB_PASSWORD", "envpass")
        monkeypatch.setenv("DB_HOST", "env-host.internal")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("DB_NAME", "mydb")
        monkeypatch.delenv("DB_HOST_RO", raising=False)

        saved = _remove_pytest_from_modules()
        try:
            with patch("model_engine_server.db.base.infra_config") as mock_infra_config:
                mock_infra = MagicMock()
                mock_infra.cloud_provider = "gcp"
                mock_infra.env = "test"
                mock_infra_config.return_value = mock_infra

                from model_engine_server.db.base import get_engine_url

                result = get_engine_url(read_only=False)

        finally:
            _restore_pytest_module(saved)

        assert result.url == "postgresql://envuser:envpass@env-host.internal:5433/mydb"

    def test_gcp_without_secret_manager_prefers_db_host_ro(self, monkeypatch):
        """GCP + no DB_SECRET_NAME + DB_HOST_RO set → uses DB_HOST_RO."""
        monkeypatch.delenv("ML_INFRA_DATABASE_URL", raising=False)
        monkeypatch.delenv("DB_SECRET_NAME", raising=False)
        monkeypatch.setenv("DB_USER", "envuser")
        monkeypatch.setenv("DB_PASSWORD", "envpass")
        monkeypatch.setenv("DB_HOST", "primary.internal")
        monkeypatch.setenv("DB_HOST_RO", "replica.internal")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_NAME", "llm_engine")

        saved = _remove_pytest_from_modules()
        try:
            with patch("model_engine_server.db.base.infra_config") as mock_infra_config:
                mock_infra = MagicMock()
                mock_infra.cloud_provider = "gcp"
                mock_infra.env = "test"
                mock_infra_config.return_value = mock_infra

                from model_engine_server.db.base import get_engine_url

                result = get_engine_url(read_only=True)

        finally:
            _restore_pytest_module(saved)

        assert result.url == "postgresql://envuser:envpass@replica.internal:5432/llm_engine"
