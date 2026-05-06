import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("value", ["0", "-1"])
def test_model_cache_lock_stale_seconds_rejects_non_positive_values(value):
    env = os.environ.copy()
    env["GIT_TAG"] = "test"
    env["MODEL_CACHE_LOCK_STALE_SECONDS"] = value

    result = subprocess.run(
        [sys.executable, "-c", "import model_engine_server.common.env_vars"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "MODEL_CACHE_LOCK_STALE_SECONDS must be a positive integer" in result.stderr
