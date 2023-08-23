import threading
import time

import pytest
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.inference.forwarding.http_forwarder import (
    MultiprocessingConcurrencyLimiter,
    predict,
)


class ExceptionCapturedThread(threading.Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)
        self.ex = None

    def run(self):
        try:
            self._target(*self._args)
        except Exception as e:
            self.ex = e

    def join(self):
        super().join()
        if self.ex is not None:
            raise self.ex


def mock_forwarder(dict):
    time.sleep(1)
    return dict


def test_http_service_429():
    limiter = MultiprocessingConcurrencyLimiter(1, True)
    t1 = ExceptionCapturedThread(
        target=predict, args=(EndpointPredictV1Request(), mock_forwarder, limiter)
    )
    t2 = ExceptionCapturedThread(
        target=predict, args=(EndpointPredictV1Request(), mock_forwarder, limiter)
    )
    t1.start()
    t2.start()
    t1.join()
    with pytest.raises(Exception):  # 429 thrown
        t2.join()
