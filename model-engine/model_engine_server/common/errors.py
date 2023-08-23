class HTTP429Exception(Exception):
    pass


class UpstreamHTTPSvcError(Exception):
    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content
