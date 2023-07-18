# Self Hosting

We provide a Helm chart that deploys LLM Engine to an Elastic Kubernetes Cluster (https://aws.amazon.com/eks/). This Helm chart should be configured to connect to dependencies (such as a postgresql database) that you may already have available in your environment.

The only portions of the helm chart that are production ready are the parts that configure and manage LLM Server itself—not Postgres, IAM, etc.

We first go over required AWS Dependencies that are required to exist before we can run `helm install` in your EKS cluster.

## AWS Dependencies 

This section describes assumptions about existing AWS resources required run to the LLMEngine Server

### EKS 
The LLM Engine server must be deployed in an EKS cluster environment. Currently only versions `1.23+` are supported. Below are the assumed requirements for the EKS cluster: 

Creating node groups
One will need to provision EKS node groups to schedule model pods. 
- "k8s.amazonaws.com/accelerator"= "nvidia-ampere-a10"
- node-lifecycle: normal
- Required node taints
- Gpu taints - [{ key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }]


### Postgresql

The LLMEngine server requires a PostgreSQL database to back data. LLM Engine currently supports postgres version 14.
Create a Postgresql database (we use AWS RDS Postgresql) if you do not have an existing one you wish to connect LLM Engine to. 

To enable LLM Engine to connect to the postgres engine, we create a kubernetes secret with the postgres url. An example yaml is provided below:

```
apiVersion: v1
kind: Secret
metadata:
  name: llm-engine-pg  # this name will be an input to our Helm Chart
data:
    database_url = "postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]"
```


### Amazon ElastiCache for Redis

The LLMEngine server requires Redis for various caching/queue functionality

## Amazon VPC

You will need to have a VPC with the following configuration:

- CIDR block: `10.53.0.0/16`
- Subnets: At least two private subnets for the RDS and ElastiCache resources

### Amazon S3

You will need to have an S3 bucket for the `launch` deployment. The ARN of this bucket should be provided to the `launch` IAM role.

### Amazon SQS

You will need to have SQS queues in your account. The `launch` IAM role should have full access to these queues.

### Amazon ECR

You will need to have ECR repositories in your account. The `launch` IAM role should have read access to these repositories.

### Identity and Access Management (IAM)

The LLMEngine server will required IAM permissions to run some operations (e.g endpoint creation)

IAM roles are used to grant permissions to AWS services. The following IAM role is required:

- **Role Name**: `k8s-main-launch-${region}`

This role requires the following permissions:

| Action | Resource |
| --- | --- |
| `s3:Get*`, `s3:List*`, `s3:Put*` | `${s3_bucket_arn}/*` |
| `sqs:*` | `arn:aws:sqs:${region}:${scale_account_id}:*` |
| `ecr:BatchGetImage`, `ecr:DescribeImages`, `ecr:GetDownloadUrlForLayer`, `ecr:ListImages` | `*` |

# Helm Chart
Now that all dependencies have been installed and configured, we can run the provided Helm chart. The values in the Helm chart will need to correspond with the resources described in the Dependencies section. 

TODO: insert Table with values