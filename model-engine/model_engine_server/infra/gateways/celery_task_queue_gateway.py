import json
import time
from typing import Any, Dict, List, Optional

import botocore
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    GetAsyncTaskV1Response,
    TaskStatus,
)
from model_engine_server.core.celery import TaskVisibility, celery_app
from model_engine_server.core.celery.app import get_redis_instance, get_redis_endpoint
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.tracing.tracing_gateway import TracingGateway
from model_engine_server.domain.exceptions import InvalidRequestException
from model_engine_server.domain.gateways.task_queue_gateway import TaskQueueGateway

logger = make_logger(logger_name())
backend_protocol = "abs" if infra_config().cloud_provider == "azure" else "s3"

celery_redis = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
    backend_protocol=backend_protocol,
)
celery_redis_24h = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
    task_visibility=TaskVisibility.VISIBILITY_24H,
    backend_protocol=backend_protocol,
)
celery_sqs = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.SQS.value),
    backend_protocol=backend_protocol,
)
celery_servicebus = celery_app(
    None, broker_type=str(BrokerType.SERVICEBUS.value), backend_protocol=backend_protocol
)


class CeleryTaskQueueGateway(TaskQueueGateway):
    def __init__(self, broker_type: BrokerType, tracing_gateway: TracingGateway):
        self.broker_type = broker_type
        assert self.broker_type in [
            BrokerType.SQS,
            BrokerType.REDIS,
            BrokerType.REDIS_24H,
            BrokerType.SERVICEBUS,
        ]
        self.tracing_gateway = tracing_gateway
        
        # Log initialization
        logger.info(
            "Initializing CeleryTaskQueueGateway",
            extra={
                "broker_type": self.broker_type.value,
                "backend_protocol": backend_protocol,
                "infra_config_cloud_provider": infra_config().cloud_provider,
            }
        )

    def _get_celery_dest(self):
        if self.broker_type == BrokerType.SQS:
            return celery_sqs
        elif self.broker_type == BrokerType.REDIS_24H:
            return celery_redis_24h
        elif self.broker_type == BrokerType.REDIS:
            return celery_redis
        else:
            return celery_servicebus

    def _log_broker_details(self, celery_dest, queue_name: str):
        """Log detailed broker connection information for debugging"""
        try:
            broker_url = celery_dest.conf.broker_url
            backend_url = celery_dest.conf.result_backend
            
            logger.info(
                "Celery broker details",
                extra={
                    "broker_type": self.broker_type.value,
                    "broker_url": broker_url,
                    "backend_url": backend_url,
                    "queue_name": queue_name,
                    "celery_app_name": celery_dest.main,
                }
            )
            
            # For Redis, also check the actual connection
            if self.broker_type in [BrokerType.REDIS, BrokerType.REDIS_24H]:
                try:
                    redis_endpoint = get_redis_endpoint(0)  # Default db
                    logger.info(
                        "Redis connection details", 
                        extra={
                            "redis_endpoint": redis_endpoint,
                            "queue_name": queue_name,
                        }
                    )
                    
                    # Test Redis connection and queue inspection
                    redis_client = get_redis_instance(0)
                    queue_length_before = redis_client.llen(queue_name)
                    redis_client.close()
                    
                    logger.info(
                        "Pre-send queue state",
                        extra={
                            "queue_name": queue_name,
                            "queue_length_before_send": queue_length_before,
                        }
                    )
                    
                except Exception as e:
                    logger.warning(
                        "Failed to inspect Redis queue state",
                        extra={
                            "queue_name": queue_name,
                            "error": str(e),
                        }
                    )
        except Exception as e:
            logger.warning(
                "Failed to log broker details",
                extra={
                    "broker_type": self.broker_type.value,
                    "error": str(e),
                }
            )

    def _verify_task_enqueued(self, queue_name: str, task_id: str):
        """Verify the task actually made it to the queue"""
        if self.broker_type in [BrokerType.REDIS, BrokerType.REDIS_24H]:
            try:
                redis_client = get_redis_instance(0)
                queue_length_after = redis_client.llen(queue_name)
                
                # Check if our task ID appears in the queue
                queue_contents = redis_client.lrange(queue_name, -5, -1)  # Last 5 items
                task_found_in_queue = any(task_id.encode() in item for item in queue_contents)
                
                redis_client.close()
                
                logger.info(
                    "Post-send queue verification",
                    extra={
                        "queue_name": queue_name,
                        "task_id": task_id,
                        "queue_length_after_send": queue_length_after,
                        "task_found_in_queue": task_found_in_queue,
                        "last_queue_items_count": len(queue_contents),
                    }
                )
                
                if not task_found_in_queue:
                    logger.warning(
                        "Task ID not found in queue after sending!",
                        extra={
                            "queue_name": queue_name,
                            "task_id": task_id,
                            "queue_length": queue_length_after,
                        }
                    )
                    
            except Exception as e:
                logger.warning(
                    "Failed to verify task enqueuing",
                    extra={
                        "queue_name": queue_name,
                        "task_id": task_id,
                        "error": str(e),
                    }
                )

    def send_task(
        self,
        task_name: str,
        queue_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        expires: Optional[int] = None,
    ) -> CreateAsyncTaskV1Response:
        send_start_time = time.time()
        
        # Log detailed send attempt
        logger.info(
            "Starting task send operation",
            extra={
                "task_name": task_name,
                "queue_name": queue_name,
                "broker_type": self.broker_type.value,
                "args_provided": args is not None,
                "kwargs_provided": kwargs is not None,
                "args_count": len(args) if args else 0,
                "kwargs_keys": list(kwargs.keys()) if kwargs else [],
                "expires": expires,
            }
        )
        
        # Used for both endpoint infra creation and async tasks
        celery_dest = self._get_celery_dest()
        kwargs = kwargs or {}
        
        # Log broker details for debugging
        self._log_broker_details(celery_dest, queue_name)
        
        with self.tracing_gateway.create_span("send_task_to_queue") as span:
            kwargs.update(self.tracing_gateway.encode_trace_kwargs())
            
            try:
                logger.info(
                    "Calling celery send_task",
                    extra={
                        "task_name": task_name,
                        "queue_name": queue_name,
                        "final_kwargs_keys": list(kwargs.keys()),
                    }
                )
                
                res = celery_dest.send_task(
                    name=task_name,
                    args=args,
                    kwargs=kwargs,
                    queue=queue_name,
                )
                
                send_duration = time.time() - send_start_time
                
                span.input = {
                    "queue_name": queue_name,
                    "args": json.loads(json.dumps(args, indent=4, sort_keys=True, default=str)),
                    "task_id": res.id,
                    "task_name": task_name,
                }
                span.output = {"task_id": res.id}
                
                logger.info(
                    "Task sent successfully",
                    extra={
                        "task_id": res.id,
                        "task_name": task_name,
                        "queue_name": queue_name,
                        "send_duration_seconds": send_duration,
                        "broker_type": self.broker_type.value,
                        "task_state": res.state,
                    }
                )
                
                # Verify the task actually made it to the queue
                self._verify_task_enqueued(queue_name, res.id)
                
            except botocore.exceptions.ClientError as e:
                send_duration = time.time() - send_start_time
                logger.error(
                    "ClientError sending task to queue",
                    extra={
                        "queue_name": queue_name,
                        "task_name": task_name,
                        "broker_type": self.broker_type.value,
                        "error_code": getattr(e, 'response', {}).get('Error', {}).get('Code'),
                        "error_message": str(e),
                        "send_duration_seconds": send_duration,
                    }
                )
                raise InvalidRequestException(f"Error sending celery task: {e}")
            except Exception as e:
                send_duration = time.time() - send_start_time
                logger.error(
                    "Unexpected error sending task to queue",
                    extra={
                        "queue_name": queue_name,
                        "task_name": task_name,
                        "broker_type": self.broker_type.value,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "send_duration_seconds": send_duration,
                    }
                )
                raise
                
            logger.info(
                f"Task {res.id} sent to queue {queue_name} from gateway"
            )  # Keep original log for compatibility
            
            return CreateAsyncTaskV1Response(task_id=res.id)

    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        # Only used for async tasks
        celery_dest = self._get_celery_dest()
        res = celery_dest.AsyncResult(task_id)
        response_state = res.state
        if response_state == "SUCCESS":
            # No longer wrapping things in the result itself, since the DTO already has a 'result' key:
            # result_dict = (
            #    response_result if type(response_result) is dict else {"result": response_result}
            # )
            status_code = None
            result = res.result
            if isinstance(result, dict) and "status_code" in result:
                # Filter out status code from result if it was added by the forwarder
                # This is admittedly kinda hacky and would technically introduce an edge case
                # if we ever decide not to have async tasks wrap response.
                status_code = result["status_code"]
                del result["status_code"]
            return GetAsyncTaskV1Response(
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                result=result,
                status_code=status_code,
            )

        elif response_state == "FAILURE":
            return GetAsyncTaskV1Response(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                traceback=res.traceback,
                status_code=None,  # probably
            )
        elif response_state == "RETRY":
            # Backwards compatibility, otherwise we'd need to add "RETRY" to the clients
            response_state = "PENDING"

        try:
            task_status = TaskStatus(response_state)
            return GetAsyncTaskV1Response(task_id=task_id, status=task_status)
        except ValueError:
            logger.info(f"Task {task_id} has an unknown state: <{response_state}> ")
            return GetAsyncTaskV1Response(task_id=task_id, status=TaskStatus.UNDEFINED)
