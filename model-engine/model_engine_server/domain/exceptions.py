from model_engine_server.core.domain_exceptions import DomainException


class ExistingEndpointOperationInProgressException(DomainException):
    """
    Thrown when a user tries to edit an endpoint that has an edit in progress
    """

    def __init__(self, message):
        self.message = message


class EndpointDeleteFailedException(DomainException):
    """
    Thrown if the server failed to delete an endpoint for whatever reason. Indicates a bug serverside
    """


class EndpointUnsupportedInferenceTypeException(DomainException):
    """
    Thrown if the requested inference type is unsupported by the endpoint.
    """


class EndpointResourceInvalidRequestException(DomainException):
    """
    Thrown if the endpoint resource requests are invalid.
    """


class EndpointInfraStateNotFound(DomainException):
    """
    Thrown if the endpoint infra_state field is expected to be not None but found to be None.
    """


class EndpointResourceInfraException(DomainException):
    """
    Thrown if the endpoint resource request passes validation, but failed for unhandled reasons.
    This corresponds to a 503 error and requires investigation by the Launch team.
    """


class EndpointLabelsException(DomainException):
    """
    Thrown if the endpoint required labels are missing or wrong.
    """


class EndpointBillingTagsMalformedException(DomainException):
    """
    Thrown if endpoint billing tags are malformed (i.e. wrong type, wrong keys, etc.)
    """


class TooManyRequestsException(DomainException):
    """
    Thrown if an endpoint returns a 429 exception for too many requests.
    """


class CorruptRecordInfraStateException(DomainException):
    """
    Thrown if the data from existing state (i.e. the db, k8s, etc.) is somehow uninterpretable
    by the code. This can occur if the state isn't being written to correctly, if we've missed
    a migration somewhere, etc.
    """


class UpstreamServiceError(DomainException):
    """
    Thrown to relay an upstream HTTP service error to the user.
    """

    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content


class LLMFineTuningMethodNotImplementedException(DomainException):
    """
    Thrown if the requested fine-tuning model/method pair is not implemented.
    """


class LLMFineTuningQuotaReached(DomainException):
    """
    Thrown if the user has run too many fine-tunes.
    """


class InvalidRequestException(DomainException):
    """
    Thrown if the user request is invalid.
    """


class CronSyntaxException(DomainException):
    """
    Thrown if the requested cron schedule has invalid syntax.
    """


class TriggerNameAlreadyExistsException(DomainException):
    """
    Thrown if the requested name already exists in the trigger repository
    """
