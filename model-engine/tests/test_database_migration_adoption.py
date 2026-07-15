from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from unittest.mock import Mock, mock_open, patch

import pytest

VERSIONS_DIR = (
    Path(__file__).parents[1] / "model_engine_server" / "db" / "migrations" / "alembic" / "versions"
)

INITIAL_MIGRATION = "2024_09_09_1736-fa3267c80731_initial.py"

ADD_COLUMN_MIGRATIONS = [
    (
        "2024_09_09_1831-b574e9711e35_chat_completion_add_extra_routes.py",
        [("bundles", "runnable_image_extra_routes")],
    ),
    (
        "2024_09_24_1456-f55525c81eb5_multinode_bundle.py",
        [
            ("bundles", "runnable_image_worker_command"),
            ("bundles", "runnable_image_worker_env"),
        ],
    ),
    (
        "2025_09_16_1741-e580182d6bfd_add_passthrough_forwarder.py",
        [("bundles", "runnable_image_forwarder_type")],
    ),
    (
        "2025_09_25_1940-221aa19d3f32_add_routes_column.py",
        [("bundles", "runnable_image_routes")],
    ),
    (
        "2026_02_10_1920-62da4f8b3403_add_task_expires_seconds_column.py",
        [("endpoints", "task_expires_seconds")],
    ),
    (
        "2026_02_20_1200-a1b2c3d4e5f6_add_queue_message_timeout_seconds_column.py",
        [("endpoints", "queue_message_timeout_seconds")],
    ),
    (
        "2026_06_16_1200-c4d5e6f7a8b9_add_status_reason_column.py",
        [("endpoints", "status_reason")],
    ),
]


def load_migration(filename: str) -> Any:
    path = VERSIONS_DIR / filename
    spec = spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_migration_adopts_an_existing_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    migration = load_migration(INITIAL_MIGRATION)
    inspector = Mock()
    inspector.has_table.return_value = True
    operation = Mock()
    monkeypatch.setattr(migration.sa, "inspect", Mock(return_value=inspector))
    monkeypatch.setattr(migration, "op", operation)

    with patch("builtins.open", mock_open()) as open_file:
        migration.upgrade()

    inspector.has_table.assert_called_once_with("endpoints", schema="hosted_model_inference")
    open_file.assert_not_called()
    operation.execute.assert_not_called()


def test_initial_migration_initializes_an_empty_database(monkeypatch: pytest.MonkeyPatch) -> None:
    migration = load_migration(INITIAL_MIGRATION)
    inspector = Mock()
    inspector.has_table.return_value = False
    operation = Mock()
    monkeypatch.setattr(migration.sa, "inspect", Mock(return_value=inspector))
    monkeypatch.setattr(migration, "op", operation)

    with patch("builtins.open", mock_open(read_data="SELECT 1;")) as open_file:
        migration.upgrade()

    open_file.assert_called_once_with(migration.INITIAL_MIGRATION_PATH)
    operation.execute.assert_called_once_with("SELECT 1;")


@pytest.mark.parametrize(("filename", "columns"), ADD_COLUMN_MIGRATIONS)
def test_upgrade_skips_columns_that_already_exist(
    filename: str,
    columns: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = load_migration(filename)
    inspector = Mock()
    inspector.get_columns.return_value = [{"name": column} for _, column in columns]
    operation = Mock()
    monkeypatch.setattr(migration.sa, "inspect", Mock(return_value=inspector))
    monkeypatch.setattr(migration, "op", operation)

    migration.upgrade()

    operation.add_column.assert_not_called()


@pytest.mark.parametrize(("filename", "columns"), ADD_COLUMN_MIGRATIONS)
def test_upgrade_adds_only_missing_columns(
    filename: str,
    columns: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = load_migration(filename)
    inspector = Mock()
    inspector.get_columns.return_value = []
    operation = Mock()
    monkeypatch.setattr(migration.sa, "inspect", Mock(return_value=inspector))
    monkeypatch.setattr(migration, "op", operation)

    migration.upgrade()

    added_columns = [
        (call.args[0], call.args[1].name, call.kwargs["schema"])
        for call in operation.add_column.call_args_list
    ]
    assert added_columns == [(table, column, "hosted_model_inference") for table, column in columns]


@pytest.mark.parametrize(("filename", "columns"), ADD_COLUMN_MIGRATIONS)
def test_downgrade_skips_columns_that_are_already_absent(
    filename: str,
    columns: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = load_migration(filename)
    inspector = Mock()
    inspector.get_columns.return_value = []
    operation = Mock()
    monkeypatch.setattr(migration.sa, "inspect", Mock(return_value=inspector))
    monkeypatch.setattr(migration, "op", operation)

    migration.downgrade()

    operation.drop_column.assert_not_called()


@pytest.mark.parametrize(("filename", "columns"), ADD_COLUMN_MIGRATIONS)
def test_downgrade_drops_only_existing_columns(
    filename: str,
    columns: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = load_migration(filename)
    inspector = Mock()
    inspector.get_columns.return_value = [{"name": column} for _, column in columns]
    operation = Mock()
    monkeypatch.setattr(migration.sa, "inspect", Mock(return_value=inspector))
    monkeypatch.setattr(migration, "op", operation)

    migration.downgrade()

    dropped_columns = [
        (call.args[0], call.args[1], call.kwargs["schema"])
        for call in operation.drop_column.call_args_list
    ]
    assert dropped_columns == [
        (table, column, "hosted_model_inference") for table, column in columns
    ]
