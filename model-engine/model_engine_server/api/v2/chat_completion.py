import traceback
from datetime import datetime
from typing import Any

import pytz
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_external_interfaces_read_only,
    verify_authentication,
)
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.llms import (
    ChatCompletionV2Request,
    ChatCompletionV2Response,
    ChatCompletionV2ResponseItem,
    ChatCompletionV2StreamErrorChunk,
    StreamError,
    StreamErrorContent,
    TokenUsage,
)
from model_engine_server.common.dtos.llms.chat_completion import ChatCompletionV2StreamSuccessChunk
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import (
    LoggerTagKey,
    LoggerTagManager,
    logger_name,
    make_logger,
)
from model_engine_server.core.utils.timer import timer
from model_engine_server.domain.exceptions import (
    EndpointUnsupportedInferenceTypeException,
    EndpointUnsupportedRequestException,
    InvalidRequestException,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MetricMetadata
from model_engine_server.domain.use_cases.llm_model_endpoint_use_cases import (
    ChatCompletionStreamV2UseCase,
    ChatCompletionSyncV2UseCase,
)
from sse_starlette import EventSourceResponse

from .common import get_metric_metadata, record_route_call

logger = make_logger(logger_name())

chat_router_v2 = APIRouter(dependencies=[Depends(record_route_call)])


def handle_streaming_exception(
    e: Exception,
    code: int,
    message: str,
):  # pragma: no cover
    tb_str = traceback.format_exception(e)
    request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
    timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d %H:%M:%S %Z")
    structured_log = {
        "error": message,
        "request_id": str(request_id),
        "traceback": "".join(tb_str),
    }
    logger.error("Exception: %s", structured_log)
    return {
        "data": ChatCompletionV2StreamErrorChunk(
            request_id=str(request_id),
            error=StreamError(
                status_code=code,
                content=StreamErrorContent(
                    error=message,
                    timestamp=timestamp,
                ),
            ),
        ).model_dump_json(exclude_none=True)
    }


async def handle_stream_request(
    external_interfaces: ExternalInterfaces,
    background_tasks: BackgroundTasks,
    request: ChatCompletionV2Request,
    auth: User,
    model_endpoint_name: str,
    metric_metadata: MetricMetadata,
):  # pragma: no cover
    use_case = ChatCompletionStreamV2UseCase(
        model_endpoint_service=external_interfaces.model_endpoint_service,
        llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
        tokenizer_repository=external_interfaces.tokenizer_repository,
    )

    with timer() as use_case_timer:
        try:
            response = await use_case.execute(
                user=auth, model_endpoint_name=model_endpoint_name, request=request
            )

            # We fetch the first response to check if upstream request was successful
            # If it was not, this will raise the corresponding HTTPException
            # If it was, we will proceed to the event generator
            first_message: ChatCompletionV2StreamSuccessChunk = await response.__anext__()
        except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc
        except (
            EndpointUnsupportedInferenceTypeException,
            EndpointUnsupportedRequestException,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except ObjectHasInvalidValueException as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Internal error occurred. Our team has been notified.",
            ) from exc

        async def event_generator(timer: timer = use_case_timer):
            try:
                ttft = None
                message = None
                yield {"data": first_message.model_dump_json(exclude_none=True)}

                async for message in response:
                    if ttft is None:
                        ttft = timer.lap()
                    # if ttft is None and message.startswith("data"):
                    #     ttft = timer.lap()
                    yield {"data": message.model_dump_json(exclude_none=True)}

                if message:
                    background_tasks.add_task(
                        external_interfaces.monitoring_metrics_gateway.emit_token_count_metrics,
                        TokenUsage(
                            num_prompt_tokens=(
                                message.usage.prompt_tokens if message.usage else None
                            ),
                            num_completion_tokens=(
                                message.usage.completion_tokens if message.usage else None
                            ),
                            total_duration=timer.duration,
                        ),
                        metric_metadata,
                    )

            # The following two exceptions are only raised after streaming begins, so we wrap the exception within a Response object
            except InvalidRequestException as exc:
                yield handle_streaming_exception(exc, 400, str(exc))
            except UpstreamServiceError as exc:
                request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
                logger.exception(
                    f"Upstream service error for request {request_id}. Error detail: {str(exc.content)}"
                )
                yield handle_streaming_exception(
                    exc,
                    500,
                    f"Upstream service error for request_id {request_id}",
                )
            except Exception as exc:
                yield handle_streaming_exception(
                    exc, 500, "Internal error occurred. Our team has been notified."
                )

        return EventSourceResponse(event_generator(timer=use_case_timer))


async def handle_sync_request(
    external_interfaces: ExternalInterfaces,
    request: ChatCompletionV2Request,
    background_tasks: BackgroundTasks,
    auth: User,
    model_endpoint_name: str,
    metric_metadata: MetricMetadata,
):
    try:
        use_case = ChatCompletionSyncV2UseCase(
            model_endpoint_service=external_interfaces.model_endpoint_service,
            llm_model_endpoint_service=external_interfaces.llm_model_endpoint_service,
            tokenizer_repository=external_interfaces.tokenizer_repository,
        )
        with timer() as use_case_timer:
            response = await use_case.execute(
                user=auth, model_endpoint_name=model_endpoint_name, request=request
            )

        background_tasks.add_task(
            external_interfaces.monitoring_metrics_gateway.emit_token_count_metrics,
            TokenUsage(
                num_prompt_tokens=(response.usage.prompt_tokens if response.usage else None),
                num_completion_tokens=(
                    response.usage.completion_tokens if response.usage else None
                ),
                total_duration=use_case_timer.duration,
            ),
            metric_metadata,
        )
        return response
    except UpstreamServiceError as exc:
        request_id = LoggerTagManager.get(LoggerTagKey.REQUEST_ID)
        logger.exception(
            f"Upstream service error for request {request_id}. Error detail: {str(exc.content)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Upstream service error for request_id {request_id}",
        )
    except (ObjectNotFoundException, ObjectNotAuthorizedException) as exc:
        if isinstance(exc, ObjectNotAuthorizedException):  # pragma: no cover
            logger.info(
                f"POST /completions-sync to endpoint {model_endpoint_name} for {auth} failed with authz error {exc.args}"
            )

        raise HTTPException(
            status_code=404,
            detail="The specified endpoint could not be found.",
        ) from exc
    except ObjectHasInvalidValueException as exc:
        raise HTTPException(status_code=400, detail=to_error_details(exc))
    except InvalidRequestException as exc:
        raise HTTPException(status_code=400, detail=to_error_details(exc))
    except EndpointUnsupportedRequestException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Endpoint does not support request: {str(exc)}",
        ) from exc
    except EndpointUnsupportedInferenceTypeException as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported inference type: {str(exc)}",
        ) from exc


def to_error_details(exc: Exception) -> Any:
    if not exc.args or len(exc.args) == 0:
        return str(exc)
    if len(exc.args) == 1:
        return exc.args[0]
    else:
        return exc.args


@chat_router_v2.post("/chat/completions", response_model=ChatCompletionV2ResponseItem)
async def chat_completion(
    request: ChatCompletionV2Request,
    background_tasks: BackgroundTasks,
    auth: User = Depends(verify_authentication),
    external_interfaces: ExternalInterfaces = Depends(get_external_interfaces_read_only),
    metric_metadata: MetricMetadata = Depends(get_metric_metadata),
) -> ChatCompletionV2Response:  # pragma: no cover
    model_endpoint_name = request.model
    if hmi_config.sensitive_log_mode:
        logger.info(
            f"POST /v2/chat/completion ({('stream' if request.stream else 'sync')}) to endpoint {model_endpoint_name} for {auth}"
        )
    else:
        logger.info(
            f"POST /v2/chat/completion ({('stream' if request.stream else 'sync')}) with {request} to endpoint {model_endpoint_name} for {auth}"
        )

    if request.stream:
        return await handle_stream_request(
            external_interfaces=external_interfaces,
            background_tasks=background_tasks,
            request=request,
            auth=auth,
            model_endpoint_name=model_endpoint_name,
            metric_metadata=metric_metadata,
        )
    else:
        return await handle_sync_request(
            external_interfaces=external_interfaces,
            background_tasks=background_tasks,
            request=request,
            auth=auth,
            model_endpoint_name=model_endpoint_name,
            metric_metadata=metric_metadata,
        )
