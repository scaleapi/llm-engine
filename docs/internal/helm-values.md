# Helm Values Reference

**Audience:** Deployment engineers installing model engine into a customer environment.
**Purpose:** Reference for every configurable value, organized by deployment concern.
**Full chart source:** `charts/model-engine/values_sample.yaml`

---

## High-Risk Values

!!! warning "Read this before starting your installation"
    The following values have non-obvious defaults or silent failure modes. Getting them wrong causes hard-to-diagnose issues.

| Value | Default | Risk | Impact if wrong |
|---|---|---|---|
| `db.runDbMigrationScript` | `false` | HIGH | Schema not initialized on first install — model creation fails with cryptic DB errors |
| `config.values.infra.prometheus_server_address` | unset | HIGH | KEDA scale-to-zero silently broken for sync endpoints with `min_workers=0` |
| `config.values.launch.vllm_repository` | `vllm` (resolves to Scale's ECR) | HIGH | Endpoint creation succeeds but pods stay INITIALIZING — image pull fails silently |
| `celeryBrokerType` | `sqs` | HIGH | Wrong broker for non-AWS clouds — all async endpoints broken |
| `config.values.infra.cloud_provider` | `aws` | HIGH | Wrong storage/auth/registry clients loaded for non-AWS environments |

!!! note "TODO"
    Change `db.runDbMigrationScript` default to `true` in `values.yaml`.

---

## 1. Minimum Viable Config

These are the values you must set for the service to start. All other values have safe defaults. Getting any of these wrong will prevent the control plane from coming up or prevent any endpoint from being created successfully.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `tag` | string | — | Yes | LLM Engine Docker image tag to deploy |
| `image.gatewayRepository` | string | — | Yes | Docker repository for the gateway image |
| `image.builderRepository` | string | — | Yes | Docker repository for the endpoint builder image |
| `image.cacherRepository` | string | — | Yes | Docker repository for the cacher image |
| `image.forwarderRepository` | string | — | Yes | Docker repository for the forwarder image |
| `secrets.kubernetesDatabaseSecretName` | string | `llm-engine-postgres-credentials` | Yes (one of two) | Kubernetes Secret name containing `DATABASE_URL`. Mutually exclusive with `secrets.cloudDatabaseSecretName` |
| `secrets.cloudDatabaseSecretName` | string | — | Yes (one of two) | Cloud-provider secret name (e.g., AWS Secrets Manager) containing database credentials |
| `serviceAccount.annotations` | map | — | Yes | Annotations to apply to the control-plane service account. On EKS, set `eks.amazonaws.com/role-arn` |
| `config.values.infra.cloud_provider` | string | `aws` | Yes | Cloud provider: `aws`, `azure`, or `onprem` |
| `config.values.infra.k8s_cluster_name` | string | `main_cluster` | Yes | Kubernetes cluster name used for resource tagging and lookups |
| `config.values.infra.dns_host_domain` | string | `llm-engine.domain.com` | Yes | Base domain for endpoint hostnames |
| `config.values.infra.default_region` | string | `us-east-1` | Yes | Default cloud region for all resource operations |
| `config.values.infra.ml_account_id` | string | `"000000000000"` | Yes | Cloud account/subscription ID |
| `config.values.infra.docker_repo_prefix` | string | `000000000000.dkr.ecr.us-east-1.amazonaws.com` | Yes | Prefix prepended to all inference image repositories |
| `config.values.infra.redis_host` | string | — | Yes (if not using secret) | Hostname of the Redis cluster used by the inference control plane |
| `config.values.infra.s3_bucket` | string | `llm-engine` | Yes | S3 bucket (or equivalent) for storing fine-tuning artifacts and other assets |
| `config.values.launch.endpoint_namespace` | string | `llm-engine` | Yes | Kubernetes namespace where inference endpoint pods are created |
| `config.values.launch.cache_redis_aws_url` | string | — | Yes (one of three) | Full Redis URL used by the cacher. Exactly one of `cache_redis_aws_url`, `cache_redis_azure_host`, or `cache_redis_aws_secret_name` must be set |

### Minimal Working YAML

```yaml
tag: "abc123def456"

image:
  gatewayRepository: public.ecr.aws/b2z8n5q1/model-engine
  builderRepository: public.ecr.aws/b2z8n5q1/model-engine
  cacherRepository: public.ecr.aws/b2z8n5q1/model-engine
  forwarderRepository: public.ecr.aws/b2z8n5q1/model-engine
  pullPolicy: Always

secrets:
  kubernetesDatabaseSecretName: llm-engine-postgres-credentials

serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::111122223333:role/k8s-main-llm-engine

db:
  runDbMigrationScript: true  # required on first install
  runDbInitScript: false

config:
  values:
    infra:
      cloud_provider: aws
      k8s_cluster_name: my-cluster
      dns_host_domain: llm-engine.example.com
      default_region: us-east-1
      ml_account_id: "111122223333"
      docker_repo_prefix: "111122223333.dkr.ecr.us-east-1.amazonaws.com"
      redis_host: my-redis.use1.cache.amazonaws.com
      s3_bucket: my-llm-engine-bucket
    launch:
      endpoint_namespace: llm-engine
      cache_redis_aws_url: redis://my-redis.use1.cache.amazonaws.com:6379/15
      s3_file_llm_fine_tuning_job_repository: "s3://my-llm-engine-bucket/llm-ft-job-repository"
      hf_user_fine_tuned_weights_prefix: "s3://my-llm-engine-bucket/fine_tuned_weights"
      vllm_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/vllm"
      tensorrt_llm_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/tensorrt-llm"
      batch_inference_vllm_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/llm-engine/batch-infer-vllm"
```

---

## 2. Cloud-Specific Config

### AWS (Reference Configuration)

AWS is the default. The values below represent the full set of AWS-specific fields.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `config.values.infra.cloud_provider` | string | `aws` | Yes | Set to `aws` |
| `config.values.infra.default_region` | string | `us-east-1` | Yes | AWS region for ECR, SQS, and other resources |
| `config.values.infra.ml_account_id` | string | — | Yes | AWS account ID (12 digits, quoted as string) |
| `config.values.infra.docker_repo_prefix` | string | — | Yes | ECR registry prefix: `<account>.dkr.ecr.<region>.amazonaws.com` |
| `config.values.infra.redis_host` | string | — | Yes (or use secret) | ElastiCache hostname |
| `config.values.infra.redis_aws_secret_name` | string | — | No | AWS Secrets Manager secret name containing Redis connection info. Fields: `scheme`, `host`, `port`, `auth_token` (optional), `query_params` (optional) |
| `config.values.infra.s3_bucket` | string | `llm-engine` | Yes | S3 bucket name for artifacts |
| `config.values.launch.cache_redis_aws_url` | string | — | Yes (one of three) | Full Redis URL: `redis://<host>:<port>/<db>` |
| `config.values.launch.cache_redis_aws_secret_name` | string | — | Yes (one of three) | AWS Secrets Manager secret with field `cache-url` containing full Redis URL |
| `config.values.launch.sqs_profile` | string | `default` | No | AWS profile for SQS operations |
| `config.values.launch.sqs_queue_policy_template` | string | — | Yes (for async) | IAM policy template for per-endpoint SQS queues. Must grant `sqs:*` to the LLM Engine role |
| `config.values.launch.sqs_queue_tag_template` | string | — | No | JSON template for SQS queue tags |
| `celeryBrokerType` | string | `sqs` | Yes | Use `sqs` for AWS async endpoints |
| `serviceAccount.annotations."eks.amazonaws.com/role-arn"` | string | — | Yes | IRSA role ARN for the control-plane service account |

```yaml
# AWS reference config diff
config:
  values:
    infra:
      cloud_provider: aws
      default_region: us-east-1
      ml_account_id: "111122223333"
      docker_repo_prefix: "111122223333.dkr.ecr.us-east-1.amazonaws.com"
      redis_host: my-redis.use1.cache.amazonaws.com
      s3_bucket: my-llm-engine-bucket
    launch:
      cache_redis_aws_url: redis://my-redis.use1.cache.amazonaws.com:6379/15
      sqs_profile: default
      sqs_queue_policy_template: >
        {
          "Version": "2012-10-17",
          "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::111122223333:role/k8s-main-llm-engine"},
            "Action": "sqs:*",
            "Resource": "arn:aws:sqs:us-east-1:111122223333:${queue_name}"
          }]
        }

celeryBrokerType: sqs

serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::111122223333:role/k8s-main-llm-engine
```

### Azure (Diff from AWS)

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `config.values.infra.cloud_provider` | string | — | Yes | Set to `azure` |
| `config.values.infra.default_region` | string | — | Yes | Azure region (e.g., `eastus`) |
| `config.values.launch.cache_redis_azure_host` | string | — | Yes | Azure Cache for Redis hostname: `<name>.redis.cache.windows.net:6380` |
| `keyvaultName` | string | `llm-engine-keyvault` | Yes | Azure Key Vault name for secret retrieval |
| `celeryBrokerType` | string | — | Yes | Set to `elasticache` for Azure Service Bus-backed broker |

!!! warning "Azure Service Bus: broker_pool_limit"
    When using Azure Service Bus as the Celery broker, do **not** set `broker_pool_limit=0`. This was previously thought to help with connection management but actually causes idle AMQP connections to drop, resulting in 503 errors on async endpoints. The fix (removing `broker_pool_limit=0`) is tracked in commit `9deb59f1`. Leave this at the library default.

```yaml
# Azure diff
config:
  values:
    infra:
      cloud_provider: azure
      default_region: eastus
      ml_account_id: "your-subscription-id"
      docker_repo_prefix: "myregistry.azurecr.io"
    launch:
      cache_redis_azure_host: my-llm-engine-cache.redis.cache.windows.net:6380
      # Do NOT set cache_redis_aws_url for Azure

keyvaultName: my-llm-engine-keyvault
celeryBrokerType: elasticache  # Azure Service Bus-backed
```

### GCP / On-Premises (Diff from AWS)

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `config.values.infra.cloud_provider` | string | — | Yes | Set to `onprem` |
| `config.values.launch.cache_redis_onprem_url` | string | — | Yes (one of three) | Explicit Redis URL for on-prem: `redis://redis:6379/0`. Highest priority — takes precedence over all other Redis URL fields |
| `celeryBrokerType` | string | — | Yes | Set to `elasticache` to use Redis as the Celery broker instead of SQS |
| `celery_broker_type_redis` | bool | `null` | No | Alternative override flag to force Redis broker regardless of `celeryBrokerType` |

```yaml
# On-prem / GCP diff
config:
  values:
    infra:
      cloud_provider: onprem
      default_region: us-central1
      ml_account_id: "my-gcp-project"
      docker_repo_prefix: "gcr.io/my-gcp-project"
    launch:
      cache_redis_onprem_url: redis://redis.llm-engine.svc.cluster.local:6379/0

celeryBrokerType: elasticache
celery_broker_type_redis: true
```

!!! note "Cloud matrix"
    For a full per-cloud capability and limitation matrix, see [cloud-matrix.md](cloud-matrix.md).

---

## 3. GPU / Hardware Config

### Balloon Pods

Balloon pods are low-priority placeholder deployments that keep GPU nodes warm. When real inference pods need to be scheduled, they preempt the balloon pods, eliminating cold-start node provisioning time.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `balloons[].acceleratorName` | string | — | Yes | GPU type identifier. Must match node labels. Supported: `nvidia-ampere-a10`, `nvidia-ampere-a100`, `nvidia-tesla-t4`, `nvidia-hopper-h100`, `cpu` |
| `balloons[].replicaCount` | integer | `0` | Yes | Number of balloon pods to maintain for this GPU type. Set to `0` to disable warming for that type |
| `balloons[].gpuCount` | integer | `1` | No | Number of GPUs each balloon pod requests. Relevant for multi-GPU balloon pods (e.g., `gpuCount: 4` for H100 nodes) |
| `balloonConfig.reserveHighPriority` | bool | `true` | No | If `true`, only high-priority pods can preempt balloon pods. If `false`, any pod can preempt balloons, which may cause unintended disruption |
| `balloonNodeSelector` | map | `{node-lifecycle: normal}` | No | Node selector applied to all balloon pod deployments. Restricts balloons to on-demand (non-spot) nodes by default |

```yaml
balloonConfig:
  reserveHighPriority: true

balloonNodeSelector:
  node-lifecycle: normal

balloons:
  - acceleratorName: nvidia-ampere-a10
    replicaCount: 2
  - acceleratorName: nvidia-ampere-a100
    replicaCount: 1
  - acceleratorName: nvidia-hopper-h100
    replicaCount: 1
    gpuCount: 4
  - acceleratorName: nvidia-tesla-t4
    replicaCount: 0
  - acceleratorName: cpu
    replicaCount: 0
```

### Image Cache

Image caching pre-pulls large inference images onto GPU nodes so that endpoint scale-up does not spend time pulling multi-GB images. Each device entry specifies a node selector (and optional tolerations) to target a specific GPU node pool.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `imageCache.devices[].name` | string | — | Yes | Logical name for this device pool (e.g., `a10`, `h100`) |
| `imageCache.devices[].nodeSelector` | map | — | Yes | Label selector targeting nodes in this device pool |
| `imageCache.devices[].tolerations` | list | `[]` | No | Tolerations for GPU taint. Required for GPU node pools with `nvidia.com/gpu:NoSchedule` taint |

```yaml
imageCache:
  devices:
    - name: cpu
      nodeSelector:
        cpu-only: "true"
    - name: a10
      nodeSelector:
        k8s.amazonaws.com/accelerator: nvidia-ampere-a10
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
    - name: a100
      nodeSelector:
        k8s.amazonaws.com/accelerator: nvidia-ampere-a100
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
    - name: t4
      nodeSelector:
        k8s.amazonaws.com/accelerator: nvidia-tesla-t4
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
    - name: h100
      nodeSelector:
        k8s.amazonaws.com/accelerator: nvidia-hopper-h100
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
    - name: h100-1g20gb
      nodeSelector:
        k8s.amazonaws.com/accelerator: nvidia-hopper-h100-1g20gb
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
    - name: h100-3g40gb
      nodeSelector:
        k8s.amazonaws.com/accelerator: nvidia-hopper-h100-3g40gb
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
```

### Recommended Hardware Tables

These tables are used by the LLM Engine to suggest appropriate hardware configurations when a user creates an endpoint without specifying hardware. The engine selects the first matching tier from `byModelName` (exact match by model name slug), then falls back to `byGpuMemoryGb` (based on model weight size in GPU memory).

#### byGpuMemoryGb

Tiers are evaluated in ascending `gpu_memory_le` order. The first tier where the model's estimated GPU memory requirement is less than or equal to `gpu_memory_le` is selected.

| `gpu_memory_le` (GB) | CPUs | GPUs | Memory | Storage | GPU Type | `nodes_per_worker` |
|---|---|---|---|---|---|---|
| 24 | 10 | 1 | 24Gi | 80Gi | nvidia-ampere-a10 | 1 |
| 48 | 20 | 2 | 48Gi | 80Gi | nvidia-ampere-a10 | 1 |
| 96 | 40 | 4 | 96Gi | 96Gi | nvidia-ampere-a10 | 1 |
| 180 | 20 | 2 | 160Gi | 160Gi | nvidia-hopper-h100 | 1 |
| 320 | 40 | 4 | 320Gi | 320Gi | nvidia-hopper-h100 | 1 |
| 640 | 80 | 8 | 800Gi | 640Gi | nvidia-hopper-h100 | 1 |
| 640 | 80 | 8 | 800Gi | 640Gi | nvidia-hopper-h100 | 2 |

#### byModelName

Exact overrides by model name slug. Takes precedence over `byGpuMemoryGb`.

| Model Name | CPUs | GPUs | Memory | Storage | GPU Type | `nodes_per_worker` |
|---|---|---|---|---|---|---|
| llama-3-8b-instruct-262k | 20 | 2 | 40Gi | 40Gi | nvidia-hopper-h100 | 1 |
| deepseek-coder-v2 | 160 | 8 | 800Gi | 640Gi | nvidia-hopper-h100 | 1 |
| deepseek-coder-v2-instruct | 160 | 8 | 800Gi | 640Gi | nvidia-hopper-h100 | 1 |

```yaml
recommendedHardware:
  byGpuMemoryGb:
    - gpu_memory_le: 24
      cpus: 10
      gpus: 1
      memory: 24Gi
      storage: 80Gi
      gpu_type: nvidia-ampere-a10
      nodes_per_worker: 1
    - gpu_memory_le: 48
      cpus: 20
      gpus: 2
      memory: 48Gi
      storage: 80Gi
      gpu_type: nvidia-ampere-a10
      nodes_per_worker: 1
    # ... additional tiers
  byModelName:
    - name: deepseek-coder-v2-instruct
      cpus: 160
      gpus: 8
      memory: 800Gi
      storage: 640Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
```

### Control-Plane Node Selector, Tolerations, and Affinity

These apply to the LLM Engine control-plane deployments (gateway, cacher, builder) — not to inference endpoint pods.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `nodeSelector` | map | `{node-lifecycle: normal}` | No | Node selector for control-plane pods. Default pins to on-demand nodes |
| `tolerations` | list | `[]` | No | Tolerations for control-plane pods |
| `affinity` | map | `{}` | No | Affinity rules for control-plane pods |

```yaml
nodeSelector:
  node-lifecycle: normal
  kubernetes.io/arch: amd64

tolerations:
  - key: "dedicated"
    operator: "Equal"
    value: "llm-engine"
    effect: "NoSchedule"

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: llm-engine-gateway
          topologyKey: kubernetes.io/hostname
```

---

## 4. Autoscaling

### Gateway Horizontal Pod Autoscaler

The HPA governs the number of gateway replicas based on concurrent request load. This applies only to the control-plane gateway deployment, not to inference endpoint pods.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `autoscaling.horizontal.enabled` | bool | `true` | No | Enable HPA for the gateway deployment |
| `autoscaling.horizontal.minReplicas` | integer | `2` | No | Minimum number of gateway replicas |
| `autoscaling.horizontal.maxReplicas` | integer | `10` | No | Maximum number of gateway replicas |
| `autoscaling.horizontal.targetConcurrency` | integer | `50` | No | Target average concurrent requests per replica before scaling out |
| `autoscaling.vertical.enabled` | bool | `false` | No | Enable Vertical Pod Autoscaler (VPA) for control-plane deployments. Requires VPA operator installed in cluster |
| `autoscaling.prewarming.enabled` | bool | `false` | No | Enable endpoint pre-warming (reserved for future use) |

### Celery Autoscaler

For async (queue-backed) endpoints, a separate Celery autoscaler process monitors queue depth and scales inference pods accordingly.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `celery_autoscaler.enabled` | bool | `true` | No | Enable the Celery autoscaler for async endpoint scaling |
| `celery_autoscaler.num_shards` | integer | `3` | No | Number of autoscaler shard instances. More shards reduces per-shard queue-watching load at high endpoint counts |

### KEDA (Scale-to-Zero for Sync Endpoints)

KEDA enables sync/streaming endpoints to scale from zero replicas to one when the first request arrives. This is distinct from the HPA (which cannot scale below `minReplicas`).

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `keda.cooldownPeriod` | integer | `300` | No | Seconds KEDA waits after the last request before scaling a sync endpoint down to zero |
| `config.values.infra.prometheus_server_address` | string | unset | Yes (for KEDA) | Address of the Prometheus server that KEDA queries for endpoint request metrics |

!!! warning "KEDA requires Prometheus"
    `config.values.infra.prometheus_server_address` must be set for KEDA scale-to-zero to function. If it is unset, sync endpoints with `min_workers=0` will **silently** fail to scale up from zero — the endpoint will appear healthy but all requests will hang until manually scaled.

!!! note "KEDA vs HPA: mutual exclusivity"
    KEDA and HPA are **mutually exclusive** per endpoint. When an endpoint has `min_workers=0`, KEDA manages scaling from 0 to 1. Once at 1+ replicas, the HPA (if configured) takes over scaling above 1. Do not configure both KEDA and HPA to manage the same endpoint's replica range. Additionally, KEDA can only scale from **0 to 1** — it does not replace the HPA for scaling beyond 1 replica.

```yaml
autoscaling:
  horizontal:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetConcurrency: 50
  vertical:
    enabled: false

celery_autoscaler:
  enabled: true
  num_shards: 3

keda:
  cooldownPeriod: 300

config:
  values:
    infra:
      prometheus_server_address: "http://prometheus-server.istio-system.svc.cluster.local:80"
```

---

## 5. Networking

### Istio VirtualService and DestinationRule

LLM Engine uses Istio for traffic routing when `config.values.launch.istio_enabled` is `true`. The VirtualService routes external traffic to the gateway service; the DestinationRule configures connection pool and outlier detection.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `virtualservice.enabled` | bool | `true` | No | Create an Istio VirtualService for the gateway |
| `virtualservice.hostDomains` | list | `[llm-engine.domain.com]` | Yes (if enabled) | List of hostnames this VirtualService responds to. Must match your Istio gateway configuration |
| `virtualservice.gateways` | list | `[default/internal-gateway]` | Yes (if enabled) | Istio Gateway resources to attach this VirtualService to. Format: `<namespace>/<gateway-name>` |
| `virtualservice.annotations` | map | `{}` | No | Additional annotations for the VirtualService resource |
| `destinationrule.enabled` | bool | `true` | No | Create an Istio DestinationRule for the gateway service |
| `destinationrule.annotations` | map | `{}` | No | Additional annotations for the DestinationRule resource |
| `hostDomain.prefix` | string | `http://` | No | URL scheme prefix used when constructing endpoint host URLs. Set to `https://` for TLS-terminated clusters |
| `service.type` | string | `ClusterIP` | No | Kubernetes Service type for the gateway. Use `ClusterIP` with Istio; change to `LoadBalancer` only if managing ingress outside Istio |
| `service.port` | integer | `80` | No | Port exposed by the gateway Kubernetes Service |
| `config.values.launch.istio_enabled` | bool | `true` | No | Whether Istio service mesh is active. When `false`, VirtualService/DestinationRule resources are not used and direct service routing applies |

```yaml
virtualservice:
  enabled: true
  hostDomains:
    - llm-engine.example.com
  gateways:
    - default/internal-gateway

destinationrule:
  enabled: true

hostDomain:
  prefix: https://

service:
  type: ClusterIP
  port: 80

config:
  values:
    launch:
      istio_enabled: true
```

### Redis TLS and Authentication

These values control TLS and authentication for the Redis connection used by KEDA and endpoint metrics.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `redis.enableTLS` | bool | `false` | No | Enable TLS for the Redis connection. Required for Azure Cache for Redis (port 6380) and any Redis with TLS enforced |
| `redis.enableAuth` | bool | `false` | No | Enable password/token authentication for Redis. Required when the Redis cluster has AUTH configured |
| `redis.auth` | string | `null` | No | Redis AUTH password or token. Only used when `enableAuth: true`. Store this in a Kubernetes Secret rather than directly in values |
| `redis.kedaSecretName` | string | `""` | No | Name of a Kubernetes Secret containing Redis credentials for KEDA's ScaledObject. KEDA reads this directly; leave empty to use unauthenticated Redis |
| `redis.unsafeSsl` | bool | `false` | No | Skip TLS certificate verification. Use only in development environments with self-signed certificates |

```yaml
redis:
  enableTLS: true
  enableAuth: true
  auth: ""  # set via --set redis.auth=$REDIS_TOKEN or from a secret
  kedaSecretName: "keda-redis-secret"
  unsafeSsl: false
```

---

## 6. Observability

### Datadog Integration

LLM Engine supports two Datadog toggles that must both be set consistently.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `datadog.enabled` | bool | `false` | No | Mount the Datadog agent socket and inject Datadog environment variables into control-plane pods. Requires the Datadog agent DaemonSet to be running on the cluster |
| `dd_trace_enabled` | bool | `true` | No | Top-level Helm toggle that controls whether the `DD_TRACE_ENABLED` environment variable is set to `true` in control-plane containers |
| `config.values.launch.dd_trace_enabled` | bool | `false` | No | Service-config-level toggle that controls whether the application code initializes the `ddtrace` library at startup. Must match `dd_trace_enabled` to avoid partial tracing |

!!! warning "Two toggles, one feature"
    `dd_trace_enabled` (top-level) and `config.values.launch.dd_trace_enabled` are independent toggles that together control Datadog APM. Setting only one of them produces a broken state: traces may be emitted but not received, or the agent socket may be mounted but no spans generated. **Always set both to the same value.**

```yaml
datadog:
  enabled: true

dd_trace_enabled: true

config:
  values:
    launch:
      dd_trace_enabled: true
```

### Logging

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `config.values.launch.sensitive_log_mode` | bool | `false` | No | When `true`, suppresses logging of request/response payloads and other PII-containing fields. Enable in customer environments that process sensitive data |
| `debug_mode` | bool/null | `null` | No | Enables verbose debug logging across infrastructure components (gateway, cacher, builder). Produces high log volume — use only for troubleshooting |

```yaml
config:
  values:
    launch:
      sensitive_log_mode: true

debug_mode: null  # set to true only during active debugging
```

---

## 7. Security / Compliance

### Pod Security Context

The pod security context applies to all containers within a pod and controls user/group identity and filesystem permissions. Uncomment and set these values when using a hardened base image (e.g., Chainguard).

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `podSecurityContext.runAsUser` | integer | unset | No | UID to run all containers as. Chainguard images use `65532` (nonroot) |
| `podSecurityContext.runAsGroup` | integer | unset | No | GID to run all containers as |
| `podSecurityContext.runAsNonRoot` | bool | unset | No | Enforce that no container runs as UID 0. Set to `true` for all production deployments |
| `podSecurityContext.fsGroup` | integer | unset | No | GID for volume mounts. Set to match `runAsGroup` so mounted secrets and configmaps are readable |

```yaml
podSecurityContext:
  runAsUser: 65532
  runAsGroup: 65532
  runAsNonRoot: true
  fsGroup: 65532
```

### Container Security Context

The container security context applies to each individual container and controls Linux capabilities and filesystem access.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `containerSecurityContext.allowPrivilegeEscalation` | bool | unset | No | Prevent the process from gaining additional privileges via setuid/setgid. Set to `false` in all production deployments |
| `containerSecurityContext.readOnlyRootFilesystem` | bool | unset | No | Mount the container root filesystem as read-only. Set to `false` if the application writes to `/tmp` or other paths on the root fs |
| `containerSecurityContext.capabilities.drop` | list | unset | No | Linux capabilities to drop. Set to `["ALL"]` to remove all capabilities and then add back only what is needed |

```yaml
containerSecurityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: false
  capabilities:
    drop:
      - ALL
```

### Inference Pod Security (serviceTemplate)

These values apply to the inference endpoint pods created by the builder — not to the control-plane pods. They are injected into each endpoint's pod spec via the service template.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `serviceTemplate.securityContext.capabilities.drop` | list | `["all"]` | No | Linux capabilities to drop from inference containers. Default drops all capabilities |
| `serviceTemplate.mountInfraConfig` | bool | `true` | No | Mount the infra ConfigMap into inference pods. Required for the endpoint to read cloud configuration |
| `serviceTemplate.createServiceAccount` | bool | `true` | No | Create a dedicated Kubernetes ServiceAccount for inference pods in the endpoint namespace |
| `serviceTemplate.serviceAccountName` | string | `model-engine` | No | Name of the ServiceAccount created for inference pods |
| `serviceTemplate.serviceAccountAnnotations` | map | — | No | Annotations for the inference pod ServiceAccount. On EKS, set `eks.amazonaws.com/role-arn` to the inference IAM role |

```yaml
serviceTemplate:
  securityContext:
    capabilities:
      drop:
        - all
  mountInfraConfig: true
  createServiceAccount: true
  serviceAccountName: model-engine
  serviceAccountAnnotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::111122223333:role/llm-engine
    "helm.sh/hook": pre-install,pre-upgrade
    "helm.sh/hook-weight": "-2"
```

### FIPS / Federal Compliance

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `celery_enable_sha256` | bool/null | `null` | No | When `true`, forces Celery to use SHA-256 message signing instead of the default SHA-1. Required in FIPS-mode environments and any environment with federal compliance mandates (FedRAMP, IL4/IL5) |

!!! warning "Coordinated rollout required for celery_enable_sha256"
    Changing `celery_enable_sha256` requires a coordinated rollout. In-flight Celery tasks signed with SHA-1 cannot be verified by workers expecting SHA-256, and vice versa. During the transition window, drain all queues before deploying new workers. Rolling updates without draining will cause task signature verification failures and silently dropped async requests.

```yaml
celery_enable_sha256: true
```

---

## 8. Replica and Resource Tuning

### Replica Counts

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `replicaCount.gateway` | integer | `2` | No | Number of gateway replicas. Minimum 2 for production HA. Overridden by HPA when `autoscaling.horizontal.enabled: true` |
| `replicaCount.cacher` | integer | `1` | No | Number of cacher replicas. The cacher maintains a local cache of Kubernetes state (endpoint pods, services). Single replica is usually sufficient |
| `replicaCount.builder` | integer | `1` | No | Number of builder replicas. The builder handles endpoint creation and image build jobs. Single replica is usually sufficient |

### Resources

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `resources.requests.cpu` | string/integer | `2` | No | CPU request for control-plane pods (gateway, cacher, builder) |
| `resources.requests.memory` | string | unset | No | Memory request for control-plane pods |
| `resources.limits.cpu` | string/integer | unset | No | CPU limit for control-plane pods |
| `resources.limits.memory` | string | unset | No | Memory limit for control-plane pods |

```yaml
replicaCount:
  gateway: 2
  cacher: 1
  builder: 1

resources:
  requests:
    cpu: 2
    memory: 4Gi
  limits:
    cpu: 4
    memory: 8Gi
```

### Pod Disruption Budget

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `podDisruptionBudget.enabled` | bool | `true` | No | Create a PodDisruptionBudget for the gateway deployment to ensure availability during node drain and rolling updates |
| `podDisruptionBudget.minAvailable` | integer/string | `1` | No | Minimum number (or percentage) of gateway pods that must remain available during voluntary disruptions |

### Database Initialization

| Value | Type | Default (`values.yaml`) | Default (`values_sample.yaml`) | Required | Description |
|---|---|---|---|---|---|
| `db.runDbMigrationScript` | bool | `true` | `false` | Yes on first install | Run Alembic schema migrations as a pre-install/pre-upgrade Job. **Must be `true` on first install** or the database schema will not be initialized |
| `db.runDbInitScript` | bool | `false` | `false` | No | Run the database initialization script (seed data). Only needed on fresh installs that require initial seed data |

!!! warning "First install: set runDbMigrationScript: true"
    `values_sample.yaml` ships with `db.runDbMigrationScript: false`. On a brand-new install, the database schema does not exist yet. Without migrations, model creation will fail with cryptic PostgreSQL errors about missing tables. Always override this to `true` on first install. After initial migration, subsequent upgrades will apply incremental migrations automatically when set to `true`.

### Database Engine Tuning

These values tune the SQLAlchemy connection pool. Defaults are appropriate for most deployments. Increase `pool_size` and `max_overflow` only when you observe connection exhaustion errors under high gateway concurrency.

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `config.values.infra.db_engine_pool_size` | integer | `10` | No | Number of persistent connections in the SQLAlchemy connection pool per process |
| `config.values.infra.db_engine_max_overflow` | integer | `10` | No | Maximum number of connections allowed above `pool_size`. Total max connections = `pool_size + max_overflow` |
| `config.values.infra.db_engine_echo` | bool | `false` | No | Log all SQL statements. Produces extremely high log volume — use only for debugging SQL query issues |
| `config.values.infra.db_engine_echo_pool` | bool | `false` | No | Log all connection pool events (checkout, checkin, overflow). Use only for debugging connection pool exhaustion |
| `config.values.infra.db_engine_disconnect_strategy` | string | `pessimistic` | No | Strategy for detecting stale/broken connections. `pessimistic` tests the connection before each use (safe but adds a small latency). `optimistic` assumes connections are valid until proven otherwise |

```yaml
config:
  values:
    infra:
      db_engine_pool_size: 10
      db_engine_max_overflow: 10
      db_engine_echo: false
      db_engine_echo_pool: false
      db_engine_disconnect_strategy: "pessimistic"
```

### LLM Inference Image Repositories

These values specify the Docker repository paths for each supported inference backend. They are combined with `config.values.infra.docker_repo_prefix` at endpoint creation time to form the full image URI.

!!! warning "vllm_repository: always override in customer environments"
    The default value `vllm` is a short relative path that resolves to Scale's internal ECR registry when combined with Scale's `docker_repo_prefix`. In customer environments with a different registry prefix, endpoint pods will attempt to pull from a non-existent or inaccessible image path. The pods will appear to be `INITIALIZING` with no clear error. **Always set `vllm_repository` to the full repository path or a prefix-relative path that exists in your registry.**

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `config.values.launch.vllm_repository` | string | `vllm` | Yes | Repository path for vLLM inference images. This is the most commonly used inference backend |
| `config.values.launch.tensorrt_llm_repository` | string | `tensorrt-llm` | No | Repository path for TensorRT-LLM inference images |
| `config.values.launch.batch_inference_vllm_repository` | string | `llm-engine/batch-infer-vllm` | No | Repository path for batch inference images (used by batch completion endpoints) |
| `config.values.launch.tgi_repository` | string | `text-generation-inference` | No | Repository path for HuggingFace Text Generation Inference images |
| `config.values.launch.lightllm_repository` | string | `lightllm` | No | Repository path for LightLLM inference images |
| `config.values.launch.sglang_repository` | string | `null` | No | Repository path for SGLang inference images. Optional; leave unset if SGLang is not used |
| `config.values.launch.user_inference_base_repository` | string | `launch/inference` | No | Base repository for custom user-defined inference images |
| `config.values.launch.user_inference_pytorch_repository` | string | `launch/inference/pytorch` | No | Repository for custom PyTorch inference images |
| `config.values.launch.user_inference_tensorflow_repository` | string | `launch/inference/tf` | No | Repository for custom TensorFlow inference images |
| `config.values.launch.docker_image_layer_cache_repository` | string | `launch-docker-build-cache` | No | Repository used as a layer cache during Docker image builds for custom endpoints |

```yaml
config:
  values:
    launch:
      # Always override these in customer environments
      vllm_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/vllm"
      tensorrt_llm_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/tensorrt-llm"
      batch_inference_vllm_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/llm-engine/batch-infer-vllm"
      tgi_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/text-generation-inference"
      lightllm_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/lightllm"
      user_inference_base_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/launch/inference"
      user_inference_pytorch_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/launch/inference/pytorch"
      user_inference_tensorflow_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/launch/inference/tf"
      docker_image_layer_cache_repository: "111122223333.dkr.ecr.us-east-1.amazonaws.com/launch-docker-build-cache"
```

### Fine-Tuning Storage

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `config.values.launch.s3_file_llm_fine_tuning_job_repository` | string | `s3://llm-engine/llm-ft-job-repository` | Yes | S3 URI (or equivalent) where fine-tuning job artifacts (checkpoints, adapters) are stored |
| `config.values.launch.hf_user_fine_tuned_weights_prefix` | string | `s3://llm-engine/fine_tuned_weights` | Yes | S3 URI prefix for storing user-uploaded fine-tuned model weights |

### Image Pull Policy

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `image.pullPolicy` | string | `Always` | No | Kubernetes image pull policy for all control-plane images. `Always` ensures the latest tag is always pulled. Set to `IfNotPresent` to avoid redundant pulls when using immutable tags |

### AWS ConfigMap

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `aws.configMap.name` | string | `default-config` | No | Name of the Kubernetes ConfigMap containing the AWS CLI configuration |
| `aws.configMap.create` | bool | `true` | No | Whether to create the AWS ConfigMap as part of the Helm release |
| `aws.profileName` | string | `default` | No | AWS profile name to use from the ConfigMap |

### Image Builder Service Account

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `imageBuilderServiceAccount.create` | bool | `true` | No | Create a dedicated ServiceAccount for the image builder. This account needs ECR push/pull permissions |
| `imageBuilderServiceAccount.annotations` | map | — | No | Annotations for the image builder ServiceAccount. On EKS, set `eks.amazonaws.com/role-arn` to a role with ECR permissions |

```yaml
imageBuilderServiceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::111122223333:role/k8s-main-llm-engine-image-builder
```

### Miscellaneous

| Value | Type | Default | Required | Description |
|---|---|---|---|---|
| `spellbook.enabled` | bool | `false` | No | Enable Spellbook integration. Reserved for Scale internal use |
| `context` | string | `production` | No | Deployment context tag. Used for labeling and log correlation. Set to a meaningful environment name (e.g., `staging`, `production`, `customer-prod`) |
| `celery_broker_type_redis` | bool/null | `null` | No | When `true`, forces the Celery broker to use Redis regardless of the `celeryBrokerType` value. Useful for on-prem and GCP deployments where SQS is unavailable |
| `keyvaultName` | string | `llm-engine-keyvault` | No | Azure Key Vault name. Only used when `cloud_provider: azure` |
