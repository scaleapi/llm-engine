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

To enable LLM Engine to connect to the postgres engine, fill out the helm chart values with the redis database's username, password, and hostname.

### Amazon S3

You will need to have an S3 bucket for LLMEngine to store [TODO]. The ARN of this bucket should be provided in the helm chart values.

### Amazon ECR

You will need to provide an ECR repository for the deployment to store fine-tuned models. The ARN of this repository should be provided in the helm chart values.

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

TODO: insert Table with values
