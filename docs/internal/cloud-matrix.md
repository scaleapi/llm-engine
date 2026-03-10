# Cloud Support Matrix

**Last updated:** 2026-02-26
**Owner:** Platform / ML Infra

This is the canonical reference for deploying model-engine into different cloud environments. It answers the question "what do I set for cloud X?" without requiring you to cross-reference other docs. For the installation runbook and step-by-step setup, see Confluence: *Installing Model Engine*.

---

## Section 1: Support Status

| Feature | AWS | Azure | GCP | On-prem |
|---|---|---|---|---|
| Message broker | SQS ✅ | Azure Service Bus ✅ | Redis / Memorystore ⚠️ | Redis ⚠️ |
| Object storage | S3 ✅ | Azure Blob Storage ✅ | GCS ⚠️ | S3-compatible (MinIO) ⚠️ |
| Image registry | ECR ✅ | ACR ✅ | GAR ⚠️ | Custom ⚠️ |
| Redis auth | URL or Secrets Manager ✅ | RBAC token via `DefaultAzureCredential` ✅ | Standard URL ⚠️ | Standard URL ⚠️ |
| Service account auth | IRSA ✅ | Azure Workload Identity ✅ | GCP Workload Identity ⚠️ | — |
| Tested end-to-end | ✅ | ✅ (partial) | ⚠️ | ⚠️ |

**Legend:**
- ✅ Tested in production
- ⚠️ Implemented in code, not yet tested end-to-end in a real deployment
- ❌ Not implemented

---

## Section 2: Per-Cloud Configuration Reference

All configuration enters the service through two mechanisms:

1. **Helm values** — rendered into the service ConfigMap and environment variables by the chart templates.
2. **Service config YAML** (`service_config.yaml`) — mounted into pods at runtime; its path is set via the `DEPLOY_SERVICE_CONFIG_PATH` environment variable.

The `config.values` block in `values.yaml` is rendered directly into the service ConfigMap and maps to the fields of `HostedModelInferenceServiceConfig` (`model_engine_server/common/config.py`). The `infra` sub-block maps to `InfraConfig` (`model_engine_server/core/config.py`).

### AWS (Reference Configuration)

AWS is the primary supported environment. The configuration below is the complete reference; all other clouds show only what differs from this.

#### Broker

```yaml
celeryBrokerType: sqs

config:
  values:
    launch:
      sqs_profile: default
      sqs_queue_policy_template: >
        {
          "Version": "2012-10-17",
          "Id": "__default_policy_ID",
          "Statement": [
            {
              "Sid": "__owner_statement",
              "Effect": "Allow",
              "Principal": {"AWS": "arn:aws:iam::000000000000:root"},
              "Action": "sqs:*",
              "Resource": "arn:aws:sqs:us-east-1:000000000000:${queue_name}"
            },
            {
              "Effect": "Allow",
              "Principal": {"AWS": "arn:aws:iam::000000000000:role/k8s-main-llm-engine"},
              "Action": "sqs:*",
              "Resource": "arn:aws:sqs:us-east-1:000000000000:${queue_name}"
            }
          ]
        }
      sqs_queue_tag_template: >
        {
          "Spellbook-Serve-Endpoint-Id": "${endpoint_id}",
          "Spellbook-Serve-Endpoint-Name": "${endpoint_name}",
          "Spellbook-Serve-Endpoint-Created-By": "${endpoint_created_by}"
        }
```

SQS queues are created automatically per endpoint by `SQSQueueEndpointResourceDelegate`. No pre-provisioning required.

#### Object Storage

```yaml
config:
  values:
    infra:
      s3_bucket: llm-engine
      default_region: us-east-1
    launch:
      s3_file_llm_fine_tuning_job_repository: "s3://llm-engine/llm-ft-job-repository"
      hf_user_fine_tuned_weights_prefix: "s3://llm-engine/fine_tuned_weights"
      batch_inference_vllm_repository: "llm-engine/batch-infer-vllm"
```

Auth is handled by IRSA — the IAM role attached to the service account is used transparently by boto3.

#### Redis

Exactly one of the following must be set:

```yaml
config:
  values:
    launch:
      # Option A — direct URL
      cache_redis_aws_url: redis://llm-engine-prod-cache.use1.cache.amazonaws.com:6379/15

      # Option B — Secrets Manager (recommended for production)
      # Secret must contain a key "cache-url" with the full Redis URL including db number
      cache_redis_aws_secret_name: sample-prod/redis-credentials
```

Also set the matching `redis_host` under `infra` for the Celery autoscaler:

```yaml
config:
  values:
    infra:
      redis_host: llm-engine-prod-cache.use1.cache.amazonaws.com
```

#### Service Account (IRSA)

```yaml
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::000000000000:role/k8s-main-llm-engine

imageBuilderServiceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::000000000000:role/k8s-main-llm-engine-image-builder

serviceTemplate:
  createServiceAccount: true
  serviceAccountName: model-engine
  serviceAccountAnnotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::000000000000:role/llm-engine
```

#### Image Registry (ECR)

```yaml
image:
  gatewayRepository: 000000000000.dkr.ecr.us-east-1.amazonaws.com/model-engine
  builderRepository: 000000000000.dkr.ecr.us-east-1.amazonaws.com/model-engine
  cacherRepository:  000000000000.dkr.ecr.us-east-1.amazonaws.com/model-engine
  forwarderRepository: 000000000000.dkr.ecr.us-east-1.amazonaws.com/model-engine

config:
  values:
    infra:
      docker_repo_prefix: "000000000000.dkr.ecr.us-east-1.amazonaws.com"
      ml_account_id: "000000000000"
    launch:
      vllm_repository: "000000000000.dkr.ecr.us-east-1.amazonaws.com/vllm"
      tensorrt_llm_repository: "000000000000.dkr.ecr.us-east-1.amazonaws.com/tensorrt-llm"
```

#### Cloud Provider Flag

```yaml
config:
  values:
    infra:
      cloud_provider: aws
```

This is the master switch. It controls which storage client, broker, and image registry class are instantiated at runtime (see `model_engine_server/api/dependencies.py`).

---

### Azure

The following values differ from the AWS reference configuration. All other values (autoscaling, balloons, networking, etc.) remain the same.

#### Broker

```yaml
celeryBrokerType: servicebus

azure:
  servicebus_namespace: my-servicebus-namespace  # the part before .servicebus.windows.net
```

The broker URL is constructed as:

```
azureservicebus://DefaultAzureCredential@{servicebus_namespace}.servicebus.windows.net
```

Auth uses `DefaultAzureCredential`, which picks up the Workload Identity token automatically. No connection string or SAS token is needed.

!!! warning "Azure Service Bus drops idle AMQP connections after 300 seconds"
    Azure Service Bus force-closes idle AMQP connections with `amqp:connection:forced` after 300 seconds of inactivity. This manifests as a **503 error on the first inference request following an idle period** — it looks like a random backend failure, not a configuration problem, which makes it very hard to diagnose without knowing about this behavior.

    **Fix (applied in commits #765 and #767):** The Celery app now sends AMQP keepalive heartbeats every 30 seconds via `uamqp_keep_alive_interval`, uses Celery's default connection pool (limit=10) to keep connections alive between requests, and enables `broker_connection_retry` so producers reconnect transparently on stale connections.

    The keepalive interval is configurable via the `SERVICEBUS_KEEP_ALIVE_INTERVAL` environment variable (default: `30`).

    **Minimum required version:** `azure-servicebus >= 7.14.3`. Version 7.11.4 has a known regression ([azure-sdk-for-python#34212](https://github.com/Azure/azure-sdk-for-python/issues/34212)) where idle AMQP connections are not properly managed.

    If you are seeing intermittent 503s on Azure after idle periods, verify you are running a version of model-engine that includes commit `9deb59f1` or later.

#### Object Storage (Azure Blob Storage)

```yaml
azure:
  abs_account_name: mystorageaccount
  abs_container_name: llm-engine

config:
  values:
    launch:
      s3_file_llm_fine_tuning_job_repository: "az://llm-engine/llm-ft-job-repository"
      hf_user_fine_tuned_weights_prefix: "az://llm-engine/fine_tuned_weights"
```

`ABS_ACCOUNT_NAME` and `ABS_CONTAINER_NAME` are injected as environment variables by the chart templates (`_helpers.tpl`). Storage auth uses `DefaultAzureCredential` (Workload Identity).

From `model_engine_server/common/io.py`, the `open_wrapper` function switches to `BlobServiceClient` when `cloud_provider == "azure"`:

```python
client = BlobServiceClient(
    f"https://{os.getenv('ABS_ACCOUNT_NAME')}.blob.core.windows.net",
    DefaultAzureCredential(),
)
```

#### Redis

```yaml
config:
  values:
    launch:
      cache_redis_azure_host: my-redis-cache.redis.cache.windows.net:6380
```

Do **not** set `cache_redis_aws_url` or `cache_redis_aws_secret_name` for Azure. The `cache_redis_url` property in `HostedModelInferenceServiceConfig` detects `cache_redis_azure_host` and builds a `rediss://` URL using an RBAC token fetched at runtime via `DefaultAzureCredential`:

```python
username = os.getenv("AZURE_OBJECT_ID")
token = DefaultAzureCredential().get_token("https://redis.azure.com/.default")
password = token.token
return f"rediss://{username}:{password}@{self.cache_redis_azure_host}"
```

The token expiry timestamp is tracked globally and the aioredis connection pool is recreated automatically when the token expires (`get_or_create_aioredis_pool` in `dependencies.py`).

```yaml
azure:
  object_id: 00000000-0000-0000-0000-000000000000   # injected as AZURE_OBJECT_ID
  client_id: 00000000-0000-0000-0000-000000000000   # used for Workload Identity
```

#### Service Account (Azure Workload Identity)

```yaml
azure:
  identity_name: my-managed-identity
  client_id: 00000000-0000-0000-0000-000000000000

serviceAccount:
  annotations:
    azure.workload.identity/client-id: 00000000-0000-0000-0000-000000000000
```

The chart injects `AZURE_CLIENT_ID` and `AZURE_OBJECT_ID` as environment variables into all pods. KEDA uses a `TriggerAuthentication` resource with `provider: azure-workload` and the same `client_id`.

#### Secret Management (Azure Key Vault)

```yaml
keyvaultName: my-llm-engine-keyvault

azure:
  keyvault_name: my-llm-engine-keyvault  # injected as KEYVAULT_NAME env var
```

Key Vault is required for Azure deployments. Database credentials can be pulled from Key Vault via `cloudDatabaseSecretName`, or supplied as a Kubernetes secret:

```yaml
secrets:
  cloudDatabaseSecretName: my-db-secret          # Key Vault secret name
  # OR
  kubernetesDatabaseSecretName: llm-engine-postgres-credentials  # K8s secret (simpler)
```

#### Image Registry (ACR)

```yaml
image:
  gatewayRepository: myregistry.azurecr.io/model-engine
  builderRepository: myregistry.azurecr.io/model-engine
  cacherRepository:  myregistry.azurecr.io/model-engine
  forwarderRepository: myregistry.azurecr.io/model-engine

config:
  values:
    infra:
      docker_repo_prefix: "myregistry.azurecr.io"
    launch:
      vllm_repository: "myregistry.azurecr.io/vllm"
      tensorrt_llm_repository: "myregistry.azurecr.io/tensorrt-llm"
```

#### Cloud Provider Flag

```yaml
config:
  values:
    infra:
      cloud_provider: azure
```

---

### GCP

!!! warning "GCP is not tested end-to-end"
    GCP support is implemented in code (added in commit #750, `f436d25e`) but has not been validated in a real deployment. These values are derived from the source code. Treat this section as a starting point; expect to debug and iterate. When you complete a GCP deployment, update this section and write a Confluence runbook.

The following values differ from the AWS reference configuration.

#### Broker

GCP uses Redis (Google Memorystore) as the Celery broker. This is the **legacy broker path** — Redis was replaced by SQS (AWS) and Azure Service Bus (Azure) due to reliability limitations at scale. See Section 3 for known limitations.

```yaml
celery_broker_type_redis: true  # or celeryBrokerType: elasticache

redisHost: my-memorystore-instance.redis.cache.googleapis.com
redisPort: "6379"
```

The chart injects `REDIS_HOST` and `REDIS_PORT` as environment variables. `RedisQueueEndpointResourceDelegate` is selected for GCP in `dependencies.py`:

```python
elif infra_config().cloud_provider == "gcp":
    queue_delegate = RedisQueueEndpointResourceDelegate(redis_client=redis_client)
```

#### Object Storage (GCS)

```yaml
config:
  values:
    launch:
      s3_file_llm_fine_tuning_job_repository: "gs://my-bucket/llm-ft-job-repository"
      hf_user_fine_tuned_weights_prefix: "gs://my-bucket/fine_tuned_weights"
```

Storage auth uses GCP Workload Identity via Application Default Credentials (ADC). `GCSFilesystemGateway` uses `gcloud-aio-storage` and `google-cloud-storage` — no additional credential configuration is needed if Workload Identity is properly configured on the node pool.

#### Redis (GCP Memorystore)

```yaml
config:
  values:
    launch:
      cache_redis_onprem_url: redis://my-memorystore-instance.redis.cache.googleapis.com:6379/0
```

Alternatively, `cache_redis_aws_url` is also accepted (the on-prem fallback path in `config.py` accepts it with a log warning). The same Redis instance serves as both the caching layer and the Celery broker.

#### Service Account (GCP Workload Identity)

```yaml
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account: model-engine@my-project.iam.gserviceaccount.com
```

The specific annotation format depends on your GKE setup (standard Workload Identity vs. Workload Identity Federation). ADC picks up the bound service account automatically.

#### Image Registry (GAR)

```yaml
image:
  gatewayRepository: us-docker.pkg.dev/my-project/my-repo/model-engine
  builderRepository: us-docker.pkg.dev/my-project/my-repo/model-engine
  cacherRepository:  us-docker.pkg.dev/my-project/my-repo/model-engine
  forwarderRepository: us-docker.pkg.dev/my-project/my-repo/model-engine

config:
  values:
    infra:
      # Must be parseable as {location}-docker.pkg.dev/{project}/{repository}
      docker_repo_prefix: "us-docker.pkg.dev/my-project/my-repo"
    launch:
      vllm_repository: "us-docker.pkg.dev/my-project/my-repo/vllm"
      tensorrt_llm_repository: "us-docker.pkg.dev/my-project/my-repo/tensorrt-llm"
```

`GARDockerRepository` parses `docker_repo_prefix` by splitting on `/` to extract `(location, project, repository)` and calls the Artifact Registry API using ADC. Image **builds** are not supported via GAR — `build_image` raises `NotImplementedError`.

#### Cloud Provider Flag

```yaml
config:
  values:
    infra:
      cloud_provider: gcp
```

---

### On-prem

!!! warning "On-prem is not tested end-to-end"
    On-prem support is implemented in code (added in commit #744, `ecc8ff42`) but has not been validated in a full production deployment. Known limitations: Redis-only broker (no managed queue service), no managed secret storage, NVIDIA driver setup is not covered here. Treat this section as a starting point.

On-prem uses Redis as both the Celery broker and the caching layer, and S3-compatible object storage (typically MinIO).

#### Broker

On-prem uses `OnPremQueueEndpointResourceDelegate`, selected in `dependencies.py` when `cloud_provider == "onprem"`. This is the **legacy broker path** — see Section 3 for known limitations.

```yaml
celery_broker_type_redis: true  # or celeryBrokerType: elasticache

redisHost: redis.my-namespace.svc.cluster.local
redisPort: "6379"
```

#### Object Storage (S3-compatible / MinIO)

```yaml
s3EndpointUrl: http://minio.my-namespace.svc.cluster.local:9000

config:
  values:
    infra:
      s3_bucket: llm-engine
    launch:
      s3_file_llm_fine_tuning_job_repository: "s3://llm-engine/llm-ft-job-repository"
      hf_user_fine_tuned_weights_prefix: "s3://llm-engine/fine_tuned_weights"
```

`S3_ENDPOINT_URL` is injected as an environment variable when `s3EndpointUrl` is set in helm values. boto3 and smart_open pick this up automatically, redirecting all S3 calls to the MinIO endpoint.

The code in `dependencies.py` falls through to `S3FilesystemGateway` and `S3LLMArtifactGateway` for any `cloud_provider` value that is not `azure` or `gcp` — this covers both `aws` and `onprem`:

```python
else:
    # AWS uses S3, on-prem uses MinIO (S3-compatible)
    filesystem_gateway = S3FilesystemGateway()
    llm_artifact_gateway = S3LLMArtifactGateway()
```

#### Redis

```yaml
config:
  values:
    launch:
      cache_redis_onprem_url: redis://redis.my-namespace.svc.cluster.local:6379/0
```

If `cache_redis_onprem_url` is not set, the code falls back to the `REDIS_HOST` / `REDIS_PORT` environment variables, then defaults to `redis://redis:6379/0`. The same Redis instance serves as both the caching layer and the Celery broker.

#### Service Account

On-prem has no managed IAM system. No service account annotations are required if your pods access storage via static MinIO credentials or in-cluster service accounts with appropriate RBAC. Configure MinIO credentials via environment variables or a Kubernetes secret as appropriate for your setup.

#### Secret Management

On-prem has no managed secret store. Use Kubernetes secrets directly:

```yaml
secrets:
  kubernetesDatabaseSecretName: llm-engine-postgres-credentials
```

#### Image Registry (Custom)

```yaml
image:
  gatewayRepository: my-registry.internal/model-engine
  builderRepository: my-registry.internal/model-engine
  cacherRepository:  my-registry.internal/model-engine
  forwarderRepository: my-registry.internal/model-engine

config:
  values:
    infra:
      docker_repo_prefix: "my-registry.internal"
    launch:
      vllm_repository: "my-registry.internal/vllm"
      tensorrt_llm_repository: "my-registry.internal/tensorrt-llm"
```

`OnPremDockerRepository` is selected when `cloud_provider == "onprem"`. Image pulls use the standard Kubernetes image pull mechanism with whatever credentials are configured in your registry secret.

#### Cloud Provider Flag

```yaml
config:
  values:
    infra:
      cloud_provider: onprem
```

---

## Section 3: Key Behavior Differences

| Behavior | AWS | Azure | GCP | On-prem |
|---|---|---|---|---|
| Async broker | SQS — queue per endpoint, auto-created | Azure Service Bus — topic per endpoint | Redis / Memorystore | Redis |
| Queue delegate class | `SQSQueueEndpointResourceDelegate` | `ASBQueueEndpointResourceDelegate` | `RedisQueueEndpointResourceDelegate` | `OnPremQueueEndpointResourceDelegate` |
| Inference task queue | `CeleryTaskQueueGateway(BrokerType.SQS)` | `CeleryTaskQueueGateway(BrokerType.SERVICEBUS)` | `CeleryTaskQueueGateway(BrokerType.REDIS)` | `CeleryTaskQueueGateway(BrokerType.REDIS_24H)` |
| KEDA autoscaling trigger | Redis list (via `redis_host`) | Azure Service Bus queue | Redis list | Redis list |
| Object storage client | `S3FilesystemGateway` / `S3LLMArtifactGateway` | `ABSFilesystemGateway` / `ABSLLMArtifactGateway` | `GCSFilesystemGateway` / `GCSLLMArtifactGateway` | `S3FilesystemGateway` (via MinIO) |
| Storage auth | IAM / IRSA (boto3 transparent) | `DefaultAzureCredential` (Workload Identity) | ADC (GCP Workload Identity) | Static credentials / MinIO config |
| Redis auth | Plain URL or Secrets Manager | RBAC token via `DefaultAzureCredential`, refreshed on expiry | Standard URL | Standard URL |
| Secret management | AWS Secrets Manager | Azure Key Vault (`keyvaultName` required) | — (ADC only) | K8s secrets only |
| Image registry class | `ECRDockerRepository` | `ACRDockerRepository` | `GARDockerRepository` | `OnPremDockerRepository` |
| Inference autoscaling metrics | `RedisInferenceAutoscalingMetricsGateway` | `ASBInferenceAutoscalingMetricsGateway` | `RedisInferenceAutoscalingMetricsGateway` | `RedisInferenceAutoscalingMetricsGateway` |

### Azure Service Bus — Idle Connection Drop (Critical)

!!! warning "Azure Service Bus idle AMQP connections — root cause of intermittent 503s"
    Azure Service Bus force-closes idle AMQP connections after **300 seconds** with an `amqp:connection:forced` error. The symptom is a **503 on the first inference request after a quiet period**. It looks like a flaky backend failure, not a configuration problem — which makes it very hard to diagnose without knowing about this behavior.

    **Timeline of the fix:**

    - **Commit #765 (`1fefec11`):** Added `uamqp_keep_alive_interval=30` to send AMQP heartbeats, added retry policy (`retry_total=3`, `retry_backoff_factor=0.8`, `retry_backoff_max=120`), enabled `broker_connection_retry` and `broker_connection_retry_on_startup`, and set `broker_pool_limit=0` to avoid reusing connections Azure had already closed.
    - **Commit #767 (`9deb59f1`):** Reverted `broker_pool_limit=0`. Setting pool limit to 0 destroys pool connections after each request, orphaning kombu's cached `ServiceBusClient` and sender objects — meaning keepalive heartbeats can never flow. The correct fix is the **default Celery connection pool (limit=10)** combined with `uamqp_keep_alive_interval=30`. Also bumped `azure-servicebus` from `7.11.4` to `7.14.3` to fix a known SDK regression.

    **Current behavior (post #767):** Keepalive heartbeats flow every 30 seconds over pooled connections. Azure never reaches the 300s idle timeout.

    **Configuration:** The keepalive interval is overridable via the `SERVICEBUS_KEEP_ALIVE_INTERVAL` environment variable (default: `30`).

    **Minimum required dependency:** `azure-servicebus >= 7.14.3`. Version 7.11.4 has a known regression ([azure-sdk-for-python#34212](https://github.com/Azure/azure-sdk-for-python/issues/34212)) that prevents idle connection management from working correctly even with the code changes above.

    **Diagnosis:** If you see intermittent 503s on Azure after idle periods, check gateway and builder pod logs for `amqp:connection:forced`. Confirm you are running a model-engine image that includes commit `9deb59f1`.

### GCP and On-prem — Redis as Broker (Legacy Path)

!!! warning "Redis broker is the legacy path — known reliability limitations at scale"
    Both GCP and on-prem use Redis as the Celery broker. Redis was the **original broker for all clouds**, but was replaced by SQS on AWS and Azure Service Bus on Azure due to reliability and stability issues at scale.

    **Known limitations of Redis-as-broker:**

    - No dead-letter queue — failed tasks are lost unless the worker explicitly retries.
    - No per-message visibility timeout — if a worker dies mid-task, the message stays invisible until the Redis key expires or is manually cleared.
    - Queue lifecycle management is handled in-process by the queue delegate (not by a managed service), which is less reliable under high load or during pod restarts.
    - KEDA autoscaling for async endpoints uses a Redis list trigger rather than a managed queue trigger. This code path is less battle-tested.

    If you are deploying at scale on GCP, evaluate whether a managed broker (Cloud Pub/Sub, or a managed Redis with stronger delivery guarantees) is feasible before committing to the Redis broker path.

---

## Section 4: Image Mirroring Requirements

!!! warning "Image mirroring is the #1 silent deployment failure"
    Model-engine defaults to pulling images from Scale's internal ECR (`public.ecr.aws/b2z8n5q1/`). In any customer or non-Scale environment, these pulls will either fail outright or pull images that do not match your pinned version.

    **Failure mode if mirroring is skipped:** Endpoint creation returns HTTP 200. The endpoint stays in `INITIALIZING` forever. There is no clear error message surfaced by the API — the image pull failure is buried in the pod events of the inference pod, which is created by Service Builder in the endpoint namespace.

    **Always mirror images before running `helm install`.**

### Why Mirroring Is Required

The helm chart and service config have several hardcoded or defaulted image references pointing to Scale's ECR:

| Config key | Default value | What it affects |
|---|---|---|
| `image.gatewayRepository` | `public.ecr.aws/b2z8n5q1/model-engine` | Gateway pod image pull |
| `image.builderRepository` | `public.ecr.aws/b2z8n5q1/model-engine` | Builder pod image pull |
| `image.cacherRepository` | `public.ecr.aws/b2z8n5q1/model-engine` | Cacher pod image pull |
| `image.forwarderRepository` | `public.ecr.aws/b2z8n5q1/model-engine` | Forwarder container in inference pods |
| `config.values.launch.vllm_repository` | `vllm` (relative, resolves against `docker_repo_prefix`) | vLLM image for LLM endpoints |
| `config.values.launch.tensorrt_llm_repository` | `tensorrt-llm` | TensorRT-LLM image |
| `config.values.launch.batch_inference_vllm_repository` | `llm-engine/batch-infer-vllm` | Batch inference image |

### Images to Mirror

| Image name | Source path | Notes |
|---|---|---|
| `model-engine` | `public.ecr.aws/b2z8n5q1/model-engine` | Used for gateway, builder, cacher, and forwarder |
| `vllm` | `public.ecr.aws/b2z8n5q1/vllm` | Used for LLM inference endpoints |
| `tensorrt-llm` | `public.ecr.aws/b2z8n5q1/tensorrt-llm` | Used for TensorRT-LLM inference endpoints |
| `batch-infer-vllm` | `public.ecr.aws/b2z8n5q1/llm-engine/batch-infer-vllm` | Used for batch inference jobs |

### Step-by-Step Mirroring Process

#### Step 1: Authenticate to the source (Scale's public ECR)

```bash
aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin public.ecr.aws
```

#### Step 2: Pull the images

```bash
export ME_TAG=<model-engine image tag, e.g. 60ac144c55aad971cdd7f152f4f7816ce2fb7d2f>
export VLLM_TAG=<vllm image tag>
export TRT_TAG=<tensorrt-llm image tag>
export BATCH_TAG=<batch-infer-vllm image tag>

docker pull public.ecr.aws/b2z8n5q1/model-engine:${ME_TAG}
docker pull public.ecr.aws/b2z8n5q1/vllm:${VLLM_TAG}
docker pull public.ecr.aws/b2z8n5q1/tensorrt-llm:${TRT_TAG}
docker pull public.ecr.aws/b2z8n5q1/llm-engine/batch-infer-vllm:${BATCH_TAG}
```

#### Step 3: Tag and push to your registry

**AWS ECR**

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 000000000000.dkr.ecr.us-east-1.amazonaws.com

# Create repositories if they don't exist
aws ecr create-repository --repository-name model-engine --region us-east-1
aws ecr create-repository --repository-name vllm --region us-east-1
aws ecr create-repository --repository-name tensorrt-llm --region us-east-1
aws ecr create-repository --repository-name llm-engine/batch-infer-vllm --region us-east-1

export REGISTRY=000000000000.dkr.ecr.us-east-1.amazonaws.com

docker tag public.ecr.aws/b2z8n5q1/model-engine:${ME_TAG} ${REGISTRY}/model-engine:${ME_TAG}
docker push ${REGISTRY}/model-engine:${ME_TAG}

docker tag public.ecr.aws/b2z8n5q1/vllm:${VLLM_TAG} ${REGISTRY}/vllm:${VLLM_TAG}
docker push ${REGISTRY}/vllm:${VLLM_TAG}

docker tag public.ecr.aws/b2z8n5q1/tensorrt-llm:${TRT_TAG} ${REGISTRY}/tensorrt-llm:${TRT_TAG}
docker push ${REGISTRY}/tensorrt-llm:${TRT_TAG}

docker tag public.ecr.aws/b2z8n5q1/llm-engine/batch-infer-vllm:${BATCH_TAG} \
  ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
docker push ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
```

**Azure ACR**

```bash
# Authenticate
az acr login --name myregistry

export REGISTRY=myregistry.azurecr.io

docker tag public.ecr.aws/b2z8n5q1/model-engine:${ME_TAG} ${REGISTRY}/model-engine:${ME_TAG}
docker push ${REGISTRY}/model-engine:${ME_TAG}

docker tag public.ecr.aws/b2z8n5q1/vllm:${VLLM_TAG} ${REGISTRY}/vllm:${VLLM_TAG}
docker push ${REGISTRY}/vllm:${VLLM_TAG}

docker tag public.ecr.aws/b2z8n5q1/tensorrt-llm:${TRT_TAG} ${REGISTRY}/tensorrt-llm:${TRT_TAG}
docker push ${REGISTRY}/tensorrt-llm:${TRT_TAG}

docker tag public.ecr.aws/b2z8n5q1/llm-engine/batch-infer-vllm:${BATCH_TAG} \
  ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
docker push ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
```

**GCP GAR**

```bash
# Authenticate
gcloud auth configure-docker us-docker.pkg.dev

export REGISTRY=us-docker.pkg.dev/my-project/my-repo

docker tag public.ecr.aws/b2z8n5q1/model-engine:${ME_TAG} ${REGISTRY}/model-engine:${ME_TAG}
docker push ${REGISTRY}/model-engine:${ME_TAG}

docker tag public.ecr.aws/b2z8n5q1/vllm:${VLLM_TAG} ${REGISTRY}/vllm:${VLLM_TAG}
docker push ${REGISTRY}/vllm:${VLLM_TAG}

docker tag public.ecr.aws/b2z8n5q1/tensorrt-llm:${TRT_TAG} ${REGISTRY}/tensorrt-llm:${TRT_TAG}
docker push ${REGISTRY}/tensorrt-llm:${TRT_TAG}

docker tag public.ecr.aws/b2z8n5q1/llm-engine/batch-infer-vllm:${BATCH_TAG} \
  ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
docker push ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
```

**On-prem / Custom registry**

```bash
export REGISTRY=my-registry.internal

# Authenticate using your registry's mechanism (e.g. docker login my-registry.internal)

docker tag public.ecr.aws/b2z8n5q1/model-engine:${ME_TAG} ${REGISTRY}/model-engine:${ME_TAG}
docker push ${REGISTRY}/model-engine:${ME_TAG}

docker tag public.ecr.aws/b2z8n5q1/vllm:${VLLM_TAG} ${REGISTRY}/vllm:${VLLM_TAG}
docker push ${REGISTRY}/vllm:${VLLM_TAG}

docker tag public.ecr.aws/b2z8n5q1/tensorrt-llm:${TRT_TAG} ${REGISTRY}/tensorrt-llm:${TRT_TAG}
docker push ${REGISTRY}/tensorrt-llm:${TRT_TAG}

docker tag public.ecr.aws/b2z8n5q1/llm-engine/batch-infer-vllm:${BATCH_TAG} \
  ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
docker push ${REGISTRY}/llm-engine/batch-infer-vllm:${BATCH_TAG}
```

#### Step 4: Update helm values

After mirroring, set the following in your `values.yaml` before running `helm install`:

```yaml
tag: <model-engine-tag>

image:
  gatewayRepository: <your-registry>/model-engine
  builderRepository: <your-registry>/model-engine
  cacherRepository:  <your-registry>/model-engine
  forwarderRepository: <your-registry>/model-engine

config:
  values:
    infra:
      docker_repo_prefix: "<your-registry>"
    launch:
      vllm_repository: "<your-registry>/vllm"
      tensorrt_llm_repository: "<your-registry>/tensorrt-llm"
      batch_inference_vllm_repository: "<your-registry>/llm-engine/batch-infer-vllm"
```

### Verifying Image Accessibility Before Install

Confirm that images are pullable from within the cluster before running `helm install`:

```bash
# Run a one-shot pod in the endpoint namespace to verify the pull
kubectl run image-check \
  --image=<your-registry>/model-engine:<tag> \
  --restart=Never \
  --namespace=llm-engine \
  -- echo "image pulled successfully"

kubectl logs image-check --namespace=llm-engine
kubectl delete pod image-check --namespace=llm-engine
```

### Diagnosing Image Pull Failures

If an endpoint stays `INITIALIZING` and you suspect a missing image:

```bash
# List pods in the endpoint namespace
kubectl get pods -n llm-engine

# Check for image pull errors
kubectl describe pod <pod-name> -n llm-engine | grep -A 10 Events
```

Look for `ErrImagePull` or `ImagePullBackOff` in the events. If present, either the image was not mirrored or the registry credentials are not available to the inference pod's service account.

> **[TODO]** Automate the mirroring process — a script or CI job that pulls from `public.ecr.aws/b2z8n5q1/` and pushes to the customer registry, parameterized by registry URL and image tags. Tracking in the Artifacts & Versioning Confluence page.

---

## Section 5: Known Gaps

### GCP

- **Not tested end-to-end.** GCP code paths were added in commit #750 (`f436d25e`) but have not been validated in a real deployment with real infrastructure.
- `GARDockerRepository.build_image` raises `NotImplementedError` — custom image builds (user-defined model bundles) are not supported on GCP.
- There is no GCP-specific block in the helm chart analogous to the `azure:` block. GCP currently reuses the `redisHost` / `redisPort` / `s3EndpointUrl` fields that were added for on-prem. A dedicated `gcp:` values block would improve clarity.
- No GCP-specific Confluence runbook exists. When the first GCP deployment happens, write one and link it here.

### On-prem

- **Not tested end-to-end.** On-prem support was added in commit #744 (`ecc8ff42`).
- **Redis broker only.** There is no managed message broker option for on-prem. See Section 3 for known reliability limitations of Redis-as-broker at scale.
- **No managed secret store.** All secrets must be provided as Kubernetes secrets. There is no Key Vault or Secrets Manager integration path.
- **NVIDIA driver setup is not covered here.** GPU nodes must have a working NVIDIA driver and GPU operator installed and configured before `helm install`. Verify with `nvidia-smi` on a GPU node before attempting GPU endpoint creation. If driver setup is untested in your environment, do a CPU-only smoke test first.
- `OnPremDockerRepository` does not support image builds — custom image bundles are not supported on-prem.

### Azure

- **Service Bus 300s idle connection drop** — fixed in commits #765 (`1fefec11`) and #767 (`9deb59f1`). Requires `azure-servicebus >= 7.14.3` and a model-engine image that includes `9deb59f1`. Deployments on older images will see intermittent 503s after idle periods. See Section 3 for the full diagnosis and fix details.
- Tested in production at partial scale. Edge cases involving very long idle periods or concurrent token refresh races may still surface.
