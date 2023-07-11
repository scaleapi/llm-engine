NAME = "hosted-inference-sync-service"
CONCURRENCY = 1  # TODO read from env var?? what's our api
# TODO set NUM_PROCESSES to 1 for deploying large models,
# but it should be CONCURRENCY + 1 in order to support quickly returning
# 429s back upstream when the pod is at capacity
NUM_PROCESSES = 1
FAIL_ON_CONCURRENCY_LIMIT = True  # TODO read from env var??
