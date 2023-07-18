# Self Hosting

We provide a Helm chart that deploys LLM Engine to an Elastic Kubernetes Cluster (https://aws.amazon.com/eks/). This Helm chart should be configured to connect to dependencies (such as a postgresql database) that you may already have available in your environment.

The only portions of the helm chart that are production ready are the parts that configure and manage LLM Server itself—not Postgres, IAM, etc.

We first go over required AWS Dependencies that are required to exist before we can run `helm install` in your EKS cluster.

## AWS Dependencies 

This section describes assumptions about existing AWS resources required run to the LLMEngine Server

### EKS 
The LLM Engine server must be deployed in an EKS cluster environment. Currently only versions `1.23+` are supported. Below are the assumed requirements for the EKS cluster: 

One will need to provision EKS node groups with GPUs to schedule model pods. These node groups must have the `node-lifecycle: normal` label on them.
Additionally, they must have the `k8s.amazonaws.com/accelerator` label set appropriately depending on the instance type:

| Instance family | `k8s.amazonaws.com/accelerator` label |
| --- | --- |
| g4dn | nvidia-tesla-t4 |
| g5 | nvidia-tesla-a10 |
| p4d | nvidia-tesla-a100 |
| p4de | nvidia-tesla-a100e |

We also recommend setting the following taint on your GPU nodes to prevent pods requiring GPU resources from being scheduled on them:
- { key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }


### PostgreSQL

The LLMEngine server requires a PostgreSQL database to back data. LLMEngine currently supports Postgres version 14.
Create a PostgreSQL database (e.g. AWS RDS PostgreSQL) if you do not have an existing one you wish to connect LLMEngine to. 

To enable LLM Engine to connect to the postgres engine, fill out the helm chart values with the postgres database's username, password, database name, and hostname.

### Redis

The LLMEngine server requires Redis for various caching/queue functionality. LLMEngine currently supports Redis version 6.
Create a Redis cluster (e.g. AWS Elasticache for Redis) if you do not have an existing one you wish to connect LLMEngine to.

To enable LLM Engine to connect redis, fill out the helm chart values with the redis host and url.

### Amazon S3

You will need to have an S3 bucket for LLMEngine to store various assets (e.g model weigts, prediction restuls). The ARN of this bucket should be provided in the helm chart values.

### Amazon ECR

You will need to provide an ECR repository for the deployment to store model containers. The ARN of this repository should be provided in the helm chart values.

### Amazon SQS

LLMEngine utilizes Amazon SQS to keep track of jobs. LLMEngine will create and use SQS queues as needed.

### Identity and Access Management (IAM)

The LLMEngine server will an IAM role to perform various AWS operations. This role will be assumed by the serviceaccount `llm-engine` in the `launch` namespace in the EKS cluster. The ARN of this role needs to be provided to the helm chart, and the role needs to be provided the following permissions: 

| Action | Resource |
| --- | --- |
| `s3:Get*`, `s3:Put*` | `${s3_bucket_arn}/*` |
| `s3:List*` | `${s3_bucket_arn}` |
| `sqs:*` | `arn:aws:sqs:${region}:${account_id}:llm-engine-endpoint-id-*` |
| `sqs:ListQueues` | `*` |
| `ecr:BatchGetImage`, `ecr:DescribeImages`, `ecr:GetDownloadUrlForLayer`, `ecr:ListImages` | `${ecr_repository_arn}` |

# Helm Chart
Now that all dependencies have been installed and configured, we can run the provided Helm chart. The values in the Helm chart will need to correspond with the resources described in the Dependencies section. 

| Parameter | Description | Default | Required |
| --- | --- | --- | --- |
| tag | The LLM Engine docker image tag | b144dd4e5371484be1889c76e70baec375127b52 | Yes |
| context | A user-specified deployment tag | production | No |
| image.gatewayRepository | The docker repository to pull the LLM Engine gateway image from | public.ecr.aws/b2z8n5q1/llm-engine | Yes |
| image.builderRepository | The docker repository to pull the LLM Engine endpoint builder image from | public.ecr.aws/b2z8n5q1/llm-engine | Yes |
| image.cacherRepository | The docker repository to pull the LLM Engine cacher image from | public.ecr.aws/b2z8n5q1/llm-engine | Yes |
| image.forwarderRepository | The docker repository to pull the LLM Engine forwarder image from | public.ecr.aws/b2z8n5q1/llm-engine | Yes |
| image.pullPolicy | The docker image pull policy | Always | No |
| secrets.kubernetesDatabaseSecretName | The name of the secret that contains the database credentials | ml-infra-pg | No |
| serviceAccount.annotations.eks.amazonaws.com/role-arn | The ARN of the IAM role that the service account will assume | arn:aws:iam::985572151633:role/k8s-main-llm-engine | Yes |
| serviceAccount.annotations.helm.sh/hook | Helm hook annotations | pre-install,pre-upgrade | No |
| serviceAccount.annotations.helm.sh/hook-weight | Helm hook weight | "-2" | No |
| serviceAccount.namespaces | Namespaces for the service account | [] | No |
| service.type | The type of the service | ClusterIP | No |
| service.port | The port of the service | 80 | No |
| replicaCount.gateway | The amount of replica pods for the gateway deployment | 2 | No |
| replicaCount.cacher | The amount of replica pods for the cacher deployment | 1 | No |
| replicaCount.builder | The amount of replica pods for the builder deployment | 1 | No |
| replicaCount.balloonA10 | The amount of low priority pod deployment for A10 GPU nodes | 0 | No |
| replicaCount.balloonA100 | The amount of low priority pod deployment for A100 GPU nodes | 0 | No |
| replicaCount.balloonCpu | The amount of low priority pod deployment for CPU nodes | 0 | No |
| replicaCount.balloonT4 | The amount of low priority pod deployment for T4 GPU nodes | 0 | No |
| autoscaling.horizontal.enabled | Enable horizontal autoscaling | true | No |
| autoscaling.horizontal.minReplicas | Minimum number of replicas for horizontal autoscaling | 2 | No |
| autoscaling.horizontal.maxReplicas | Maximum number of replicas for horizontal autoscaling | 10 | No |
| autoscaling.horizontal.targetConcurrency | Target concurrency for horizontal autoscaling | 50 | No |
| autoscaling.vertical.enabled | Enable vertical autoscaling | false | No |
| autoscaling.prewarming.enabled | Enable prewarming for autoscaling | false | No |
| resources.requests.cpu | CPU resources request for the deployments | 2 | No |
| nodeSelector | Node selector for the deployments | {} | No |
| tolerations | Tolerations for the deployments | [] | No |
| affinity | Affinity for the deployments | {} | No |
| datadog_trace_enabled | Enable datadog tracing | false | No |
| config.values.infra.k8s_cluster_name | The name of the k8s cluster | main_cluster | Yes |
| config.values.infra.dns_host_domain | The domain name of the k8s cluster | egp-test.scale.com | Yes |
| config.values.infra.default_region | The default AWS region for various resources | us-east-1 | Yes |
| config.values.infra.ml_account_id | The AWS account ID for various resources | "692474966980" | Yes |
| config.values.infra.docker_repo_prefix | The prefix for AWS ECR repositories | "692474966980.dkr.ecr.us-east-1.amazonaws.com" | Yes |
| config.values.infra.redis_host | The hostname of the redis cluster you wish to connect | spellbook-prod-cache-001.yoibpc.0001.use1.cache.amazonaws.com | Yes |
| config.values.infra.s3_bucket | The S3 bucket you wish to connect | "scale-ml-egp-test" | Yes |
| config.values.llm_engine.endpoint_namespace | K8s namespace the endpoints will be created in | llm-engine | Yes |
| config.values.llm_engine.cache_redis_url | The full url for the redis cluster you wish to connect | redis://spellbook-prod-cache-001.yoibpc.0001.use1.cache.amazonaws.com:6379/15 | Yes |
| config.values.llm_engine.s3_file_llm_fine_tuning_job_repository | The S3 URI for the S3 bucket/key that you wish to save fine-tuned assets | "s3://scale-ml/hosted-model-inference/llm-ft-job-repository/circleci" | Yes |
| config.values.llm_engine.sqs_profile | SQS profile for asynchronous endpoints | default | No |
| config.values.llm_engine.sqs_queue_policy_template | The IAM policy template for SQS queue for async endpoints | Provided in the sample | Yes |
| config.values.llm_engine.sqs_queue_tag_template | The tag template for SQS queue for async endpoints | Provided in the sample | No |
