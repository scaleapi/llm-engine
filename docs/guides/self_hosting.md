# Self Hosting _[Experimental]_
**This guide is currently highly experimental. Instructions are subject to change as we improve support for self-hosting.**

We provide a Helm chart that deploys LLM Engine to an [Elastic Kubernetes Cluster](https://aws.amazon.com/eks/) (EKS) in [AWS](https://aws.amazon.com/). This Helm chart should be configured to connect to dependencies (such as a PostgreSQL database) that you may already have available in your environment.

The only portions of the Helm chart that are production ready are the parts that configure and manage LLM Server itself (not PostgreSQL, IAM, etc.)

We first go over required AWS dependencies that are required to exist before we can run `helm install` in your EKS cluster.

## AWS Dependencies 

This section describes assumptions about existing AWS resources required run to the LLM Engine Server

### EKS 
The LLM Engine server must be deployed in an EKS cluster environment. Currently only versions `1.23+` are supported. Below are the assumed requirements for the EKS cluster: 

You will need to provision EKS node groups with GPUs to schedule model pods. These node groups must have the `node-lifecycle: normal` label on them.
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

The LLM Engine server requires a PostgreSQL database to back data. LLM Engine currently supports PostgreSQL version 14.
Create a PostgreSQL database (e.g. AWS RDS PostgreSQL) if you do not have an existing one you wish to connect LLM Engine to.

To enable LLM Engine to connect to the PostgreSQL engine, we create a Kubernetes secret with the PostgreSQL url. An example YAML is provided below:
```
apiVersion: v1
kind: Secret
metadata:
  name: llm-engine-database-credentials  # this name will be an input to our Helm Chart
data:
    database_url = "postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]"
```

### Redis

The LLM Engine server requires Redis for various caching/queue functionality. LLM Engine currently supports Redis version 6.
Create a Redis cluster (e.g. AWS Elasticache for Redis) if you do not have an existing one you wish to connect LLM Engine to.

To enable LLM Engine to connect redis, fill out the Helm chart values with the redis host and url.

### Amazon S3

You will need to have an S3 bucket for LLM Engine to store various assets (e.g model weigts, prediction restuls). The ARN of this bucket should be provided in the Helm chart values.

### Amazon ECR

You will need to provide an ECR repository for the deployment to store model containers. The ARN of this repository should be provided in the Helm chart values.

### Amazon SQS

LLM Engine utilizes Amazon SQS to keep track of jobs. LLM Engine will create and use SQS queues as needed.

### Identity and Access Management (IAM)

The LLM Engine server will an IAM role to perform various AWS operations. This role will be assumed by the serviceaccount `llm-engine` in the `launch` namespace in the EKS cluster. The ARN of this role needs to be provided to the Helm chart, and the role needs to be provided the following permissions:

| Action | Resource |
| --- | --- |
| `s3:Get*`, `s3:Put*` | `${s3_bucket_arn}/*` |
| `s3:List*` | `${s3_bucket_arn}` |
| `sqs:*` | `arn:aws:sqs:${region}:${account_id}:llm-engine-endpoint-id-*` |
| `sqs:ListQueues` | `*` |
| `ecr:BatchGetImage`, `ecr:DescribeImages`, `ecr:GetDownloadUrlForLayer`, `ecr:ListImages` | `${ecr_repository_arn}` |

## Helm Chart
Now that all dependencies have been installed and configured, we can run the provided Helm chart. The values in the Helm chart will need to correspond with the resources described in the Dependencies section. 

Ensure that Helm V3 is installed [instructions](https://helm.sh/docs/intro/install/) and can connect to the EKS cluster. Users should be able to install the chart with `helm install llm-engine llm-engine -f llm-engine/values_sample.yaml -n <DESIRED_NAMESPACE>`.
Below are the configurations to specify in the `values_sample.yaml` file. 

| Parameter | Description | Required |
| --- | --- | --- |
| tag | The LLM Engine docker image tag | Yes |
| context | A user-specified deployment tag | No |
| image.gatewayRepository | The docker repository to pull the LLM Engine gateway image from | Yes |
| image.builderRepository | The docker repository to pull the LLM Engine endpoint builder image from | Yes |
| image.cacherRepository | The docker repository to pull the LLM Engine cacher image from | Yes |
| image.forwarderRepository | The docker repository to pull the LLM Engine forwarder image from | Yes |
| image.pullPolicy | The docker image pull policy | No |
| secrets.kubernetesDatabaseSecretName | The name of the secret that contains the database credentials | Yes |
| serviceAccount.annotations.eks.amazonaws.com/role-arn | The ARN of the IAM role that the service account will assume | Yes |
| service.type | The service configuration for the main LLM Engine server | No |
| service.port | The service configuration for the main LLM Engine server | No |
| replicaCount | The amount of replica pods for each deployment | No |
| autoscaling | The autoscaling configuration for LLM Engine server deployments | No |
| resources.requests.cpu | The k8s resources for LLM Engine server deployments | No |
| nodeSelector | The node selector for LLM Engine server deployments | No |
| tolerations | The tolerations for LLM Engine server deployments | No |
| affinity | The affinity for LLM Engine server deployments | No |
| aws.configMap.name | The AWS configurations (by configMap) for LLM Engine server deployments | No |
| aws.configMap.create | The AWS configurations (by configMap) for LLM Engine server deployments | No |
| aws.profileName | The AWS configurations (by configMap) for LLM Engine server deployments | No |
| serviceTemplate.securityContext.capabilities.drop | Additional flags for model endpoints | No |
| serviceTemplate.mountInfraConfig | Additional flags for model endpoints | No |
| config.values.infra.k8s_cluster_name | The name of the k8s cluster | Yes |
| config.values.infra.dns_host_domain | The domain name of the k8s cluster | Yes |
| config.values.infra.default_region | The default AWS region for various resources | Yes |
| config.values.infra.ml_account_id | The AWS account ID for various resources | Yes |
| config.values.infra.docker_repo_prefix | The prefix for AWS ECR repositories | Yes |
| config.values.infra.redis_host | The hostname of the redis cluster you wish to connect | Yes |
| config.values.infra.s3_bucket | The S3 bucket you wish to connect | Yes |
| config.values.llm_engine.endpoint_namespace | K8s namespace the endpoints will be created in | Yes |
| config.values.llm_engine.cache_redis_url | The full url for the redis cluster you wish to connect | Yes |
| config.values.llm_engine.s3_file_llm_fine_tuning_job_repository | The S3 URI for the S3 bucket/key that you wish to save fine-tuned assets | Yes |
| config.values.datadog_trace_enabled | Whether to enable datadog tracing, datadog must be installed in the cluster | No |

## Play With It
Once `helm install` succeeds, you can forward port 5000 from a `llm-engine` pod and test sending requests to it.

First, see a list of pods in the namespace that you performed `helm install` in:
```
$ kubectl get pods -n <NAMESPACE_YOU_INSTALL_LLM_ENGINE>
NAME                                           READY   STATUS             RESTARTS      AGE
llm-engine-668679554-9q4wj                     1/1     Running            0             18m
llm-engine-668679554-xfhxx                     1/1     Running            0             18m
llm-engine-cacher-5f8b794585-fq7dj             1/1     Running            0             18m
llm-engine-endpoint-builder-5cd6bf5bbc-sm254   1/1     Running            0             18m
llm-engine-image-cache-a10-sw4pg               1/1     Running            0             18m 
```
Note the pod name you see may be different.

Forward a port from a `llm-engine` pod:
```
$ kubectl port-forward pod/llm-engine-<REST_OF_POD_NAME> 5000:5000 -n <NAMESPACE_YOU_INSTALL_LLM_ENGINE>
```

Then, try sending a request to get LLM model endpoints for `test-user-id`. You should get a response with empty list:
```
$ curl -X GET -H "Content-Type: application/json" -u "test-user-id:" "http://localhost:5000/v1/llm/model-endpoints"
{"model_endpoints":[]}% 
```