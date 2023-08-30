from dataclasses import dataclass


class DomainException(Exception):
    """
    Base class for exceptions thrown for domain (business logic) errors.
    """


class ObjectAlreadyExistsException(DomainException):
    """
    Thrown when the user tries to create a model with a name that already exists.
    """


class ObjectNotFoundException(DomainException):
    """
    Thrown when a required object is not found, e.g. when creating a version for a nonexistent model
    """


class ObjectNotAuthorizedException(DomainException):
    """
    Thrown when a user tries to access an object they don't own.
    """


class ObjectHasInvalidValueException(DomainException, ValueError):
    """
    Thrown when a user tries to create an object with an invalid value.
    """


class ObjectNotApprovedException(DomainException):
    """
    Thrown when a required object is not approved, e.g. for a Bundle in review.
    """


@dataclass
class DockerImageNotFoundException(DomainException):
    """
    Thrown when a user tries to specify a custom Docker image that cannot be found.
    """

    repository: str
    tag: str


class DockerBuildFailedException(DomainException):
    """
    Thrown if the server failed to build a docker image.
    """


class ReadOnlyDatabaseException(DomainException):
    """
    Thrown if the server attempted to write to a read-only database.
    """

class InvalidRequestException(DomainException):
    """
    Thrown if the request sent by the user is invalid (e.x. would occur if user-passed params are invalid).
    """