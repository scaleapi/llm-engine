import time

from model_engine_server.core.utils.timer import timer


def test_timer():
    with timer() as t:
        time.sleep(0.1)
        lap_time = t.lap()
        time.sleep(0.01)
        new_lap_time = t.lap()

    assert new_lap_time >= 0.009
    assert lap_time >= 0.09
    assert t.duration >= 0.1
