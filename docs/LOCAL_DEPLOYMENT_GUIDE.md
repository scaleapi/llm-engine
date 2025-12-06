# LLM Engine Local Deployment Guide

Quick reference for running LLM Engine locally using the master controller.

## Installation

### Prerequisites

- **Python 3.10+**
- **Docker** (for Minikube/Docker Compose mode)
- **Kubectl** (for Kubernetes modes)
- **Helm** (for local Minikube deployment)
- **Minikube** OR **Docker Desktop** with Kubernetes enabled

### Setup

```bash
# Clone the repository
git clone https://github.com/scaleapi/llm-engine.git
cd llm-engine

# Install the controller
python -m pip install -e .
python engine_controller.py --action validate --mode local
```

## Deployment Modes

### 1. Docker Compose (Easiest - Development Only)

**Best for**: Python developers who want quick local testing without Kubernetes

**Requirements**:
- Docker & Docker Compose
- 8GB RAM, 20GB disk
- No GPU support

**Steps**:

```bash
# Validate prerequisites
python engine_controller.py --action validate --mode docker

# Deploy
python engine_controller.py --action deploy --mode docker

# Wait 2-3 minutes for services to start

# Check status
python engine_controller.py --action status --mode docker

# Test the API
curl -X GET http://localhost:5000/v1/llm/model-endpoints \
    -H "Content-Type: application/json" \
    -u "test-user-id:"

# Should return: {"model_endpoints":[]}

# Stop (when done)
python engine_controller.py --action cleanup --mode docker
```

**Advantages**:
- ✓ Simple to understand
- ✓ Quick startup (2-3 min)
- ✓ No Kubernetes learning curve
- ✓ Works on Windows/Mac/Linux

**Limitations**:
- ✗ Single machine only
- ✗ No GPU support
- ✗ Services not truly isolated
- ✗ Not production-like

---

### 2. Minikube (Recommended - Local Development with Kubernetes)

**Best for**: Learning Kubernetes & realistic local testing

**Requirements**:
- Minikube installed
- 8+ CPU cores, 16GB+ RAM
- 50GB+ free disk space
- Hypervisor (HyperV, VirtualBox, or KVM2)

**Steps**:

```bash
# Validate prerequisites
python engine_controller.py --action validate --mode local

# Create custom config (optional)
cat > local_config.json << EOF
{
  "minikube_driver": "hyperv",
  "minikube_cpus": 8,
  "minikube_memory_gb": 16,
  "minikube_disk_gb": 50,
  "enable_gpu": false,
  "namespace": "llm-engine"
}
EOF

# Deploy with custom config
python engine_controller.py --action deploy --mode local --config local_config.json

# Wait 5-10 minutes for full startup

# Verify all pods are running
kubectl get pods -n llm-engine

# Port forward to access locally
kubectl port-forward -n llm-engine svc/llm-engine 5000:5000

# In another terminal, test the API
curl -X GET http://localhost:5000/v1/llm/model-endpoints \
    -H "Content-Type: application/json" \
    -u "test-user-id:"

# Create an endpoint
curl -X POST http://localhost:5000/v1/llm/model-endpoints \
    -H "Content-Type: application/json" \
    -d '{
        "name": "llama-7b-test",
        "model_name": "llama-2-7b",
        "source": "hugging_face",
        "inference_framework": "text_generation_inference",
        "inference_framework_image_tag": "0.9.3",
        "num_shards": 1,
        "endpoint_type": "streaming",
        "cpus": 4,
        "memory": "10Gi",
        "storage": "10Gi",
        "gpu_type": "cpu",
        "min_workers": 1,
        "max_workers": 1,
        "per_worker": 1
    }' \
    -u "test-user-id:"

# Wait 2-3 minutes for model download and setup

# Send inference request
curl -X POST http://localhost:5000/v1/llm/completions-sync \
    -H "Content-Type: application/json" \
    -d '{
        "model_endpoint_name": "llama-7b-test",
        "prompts": ["Tell me a joke about Kubernetes"],
        "max_new_tokens": 50,
        "temperature": 0.7
    }' \
    -u "test-user-id:"

# Cleanup (delete Minikube cluster)
python engine_controller.py --action cleanup --mode local
```

**Troubleshooting**:

```bash
# Check Minikube status
minikube status

# View logs from a specific pod
kubectl logs -n llm-engine <pod-name>

# Describe a pod to see events
kubectl describe pod -n llm-engine <pod-name>

# SSH into Minikube to debug
minikube ssh

# Increase Minikube resources (recreate)
minikube delete
minikube start --cpus=12 --memory=24G --disk-size=100G
```

**Advantages**:
- ✓ Real Kubernetes experience
- ✓ Scales to cloud easily
- ✓ Service-to-service networking
- ✓ Pod isolation and networking

**Limitations**:
- ✗ Slower than Docker Compose (5-10 min startup)
- ✗ Resource intensive (16GB+ RAM)
- ✗ GPU pass-through unreliable on most systems
- ✗ Steep learning curve for non-DevOps

---

### 3. AWS EKS (Production-Grade - For deployment to cloud)

**Best for**: Production deployments or teams with AWS/Kubernetes expertise

**Requirements**:
- AWS Account with appropriate IAM permissions
- AWS CLI configured with credentials
- Kubectl and Helm installed
- $500-5000/month budget

**Prerequisites** (must be provisioned first):
- EKS cluster (v1.23+)
- RDS PostgreSQL (v14)
- ElastiCache Redis (v6)
- S3 bucket for model artifacts
- ECR repository for images
- IAM role with appropriate permissions

**Steps**:

```bash
# Create config for AWS
cat > aws_config.json << EOF
{
  "region": "us-west-2",
  "cluster_name": "my-llm-cluster",
  "eks_version": "1.27",
  "database": {
    "host": "mydb.xxx.us-west-2.rds.amazonaws.com",
    "password": "secure-password-here"
  },
  "redis": {
    "host": "myredis.xxx.ng.0001.usw2.cache.amazonaws.com"
  }
}
EOF

# Validate
python engine_controller.py --action validate --mode cloud_aws

# Deploy
python engine_controller.py --action deploy --mode cloud_aws --config aws_config.json

# Check deployment status
kubectl get pods -n llm-engine

# Port forward to test
kubectl port-forward -n llm-engine svc/llm-engine 5000:5000 &

# Test
curl -X GET http://localhost:5000/v1/llm/model-endpoints \
    -H "Content-Type: application/json" \
    -u "test-user-id:"
```

**⚠️ Before AWS Deployment**:
- [ ] Review the EXPERT_ASSESSMENT.md
- [ ] Have AWS infrastructure prepared (RDS, ElastiCache, etc.)
- [ ] Understand cost implications (~$2000+/month)
- [ ] Plan for disaster recovery
- [ ] Implement security hardening

---

## Configuration Reference

### Docker Compose Config

```json
{
  "compose_file": "docker-compose.yml",
  "project_name": "llm-engine",
  "enable_gpu": false,
  "database": {
    "host": "postgres",
    "port": 5432,
    "username": "llm_engine",
    "password": "change_me",
    "database": "llm_engine",
    "ssl_mode": "disable"
  },
  "redis": {
    "host": "redis",
    "port": 6379,
    "database": 0
  }
}
```

### Local (Minikube) Config

```json
{
  "minikube_driver": "hyperv",
  "minikube_cpus": 8,
  "minikube_memory_gb": 16,
  "minikube_disk_gb": 50,
  "enable_gpu": false,
  "namespace": "llm-engine",
  "database": {
    "host": "postgres-service",
    "port": 5432,
    "username": "llm_engine",
    "password": "minikube-password",
    "database": "llm_engine"
  },
  "redis": {
    "host": "redis-service",
    "port": 6379
  }
}
```

### AWS Config

```json
{
  "region": "us-west-2",
  "cluster_name": "llm-engine-prod",
  "eks_version": "1.27",
  "node_instance_type": "t3.large",
  "gpu_node_instance_type": "g4dn.xlarge",
  "s3_bucket": "my-llm-engine-assets",
  "ecr_repository": "my-account.dkr.ecr.us-west-2.amazonaws.com/llm-engine",
  "iam_role_arn": "arn:aws:iam::123456789:role/llm-engine-role",
  "namespace": "llm-engine",
  "database": {
    "host": "llm-engine-db.cxxxxxx.us-west-2.rds.amazonaws.com",
    "port": 5432,
    "username": "admin",
    "password": "secure-password",
    "database": "llm_engine"
  },
  "redis": {
    "host": "llm-engine-redis.xxxxx.ng.0001.usw2.cache.amazonaws.com",
    "port": 6379
  }
}
```

---

## Common Tasks

### Creating an Inference Endpoint

```bash
# First, make sure the service is running and port-forwarded

curl -X POST http://localhost:5000/v1/llm/model-endpoints \
    -H "Content-Type: application/json" \
    -d '{
        "name": "my-model",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "source": "hugging_face",
        "inference_framework": "text_generation_inference",
        "inference_framework_image_tag": "1.0.3",
        "num_shards": 1,
        "endpoint_type": "streaming",
        "cpus": 4,
        "gpus": 0,
        "memory": "16Gi",
        "storage": "20Gi",
        "gpu_type": "cpu",
        "min_workers": 1,
        "max_workers": 2,
        "per_worker": 1
    }' \
    -u "my-user-id:"

# Response will include: endpoint_creation_task_id
# Wait 5-10 minutes for the model to download and start
```

### Running Inference

```bash
curl -X POST http://localhost:5000/v1/llm/completions-sync \
    -H "Content-Type: application/json" \
    -d '{
        "model_endpoint_name": "my-model",
        "prompts": ["What is machine learning?"],
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }' \
    -u "my-user-id:"
```

### Fine-tuning a Model

```bash
curl -X POST http://localhost:5000/v1/llm/fine-tunes \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "training_data": "s3://my-bucket/training_data.jsonl",
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "num_epochs": 3
        }
    }' \
    -u "my-user-id:"

# Check fine-tuning status
curl -X GET http://localhost:5000/v1/llm/fine-tunes \
    -H "Content-Type: application/json" \
    -u "my-user-id:"
```

### Using the Python SDK

```python
from llmengine import Completion, FineTune

# Inference
response = Completion.create(
    model="my-model",  # Your deployed endpoint
    prompt="Explain machine learning in simple terms",
    max_new_tokens=100,
    temperature=0.7,
    stream=False
)
print(response.output.text)

# Streaming inference
stream = Completion.create(
    model="my-model",
    prompt="Write a poem about AI",
    max_new_tokens=200,
    stream=True
)

for response in stream:
    if response.output:
        print(response.output.text, end="", flush=True)

# Fine-tuning
fine_tune = FineTune.create(
    model="meta-llama/Llama-2-7b-hf",
    training_file="s3://bucket/data.jsonl"
)

# Check status
status = FineTune.get(fine_tune.id)
print(f"Status: {status.status}")  # queued, training, succeeded, failed
```

---

## Performance Tuning

### Docker Compose
```bash
# Increase Docker memory allocation
# Windows: Docker Desktop Settings → Resources → Memory: 12GB

# Linux: No Docker daemon limits by default
# Check with: docker info | grep Memory
```

### Minikube
```bash
# Restart with more resources
minikube delete
minikube start --cpus=12 --memory=24G --disk-size=100G

# Allocate resources to pods in values config
{
  "resources": {
    "requests": {
      "cpu": "2000m",
      "memory": "4Gi"
    },
    "limits": {
      "cpu": "4000m",
      "memory": "8Gi"
    }
  }
}
```

### AWS
```bash
# Use spot instances for cost savings (70% discount)
# Modify EKS node group to use spot instances

# Enable horizontal pod autoscaling
# Configured in Helm values: autoscaling.horizontal.enabled = true

# Use reserved instances for baseline capacity
# 1-year reserved instance = ~40% cheaper
```

---

## Debugging

### Check logs

```bash
# Docker Compose
docker-compose -f docker-compose.yml logs llm-engine
docker-compose -f docker-compose.yml logs postgres
docker-compose -f docker-compose.yml logs redis

# Minikube
kubectl logs -n llm-engine deployment/llm-engine -f
kubectl logs -n llm-engine deployment/llm-engine-cacher -f
kubectl logs -n llm-engine deployment/llm-engine-endpoint-builder -f

# AWS
kubectl logs -n llm-engine deployment/llm-engine -f --region us-west-2
```

### Check connectivity

```bash
# Test database connection
psql -h postgres-service -U llm_engine -d llm_engine

# Test Redis connection
redis-cli -h redis-service ping

# Test API endpoint
curl http://localhost:5000/health  # or /v1/health
```

### Monitor resources

```bash
# Minikube
minikube dashboard

# Kubernetes
kubectl top nodes
kubectl top pods -n llm-engine

# AWS CloudWatch
aws cloudwatch get-metric-statistics \
    --namespace AWS/EKS \
    --metric-name node_cpu_utilization \
    --start-time 2025-01-01T00:00:00Z \
    --end-time 2025-01-02T00:00:00Z \
    --period 300
```

---

## Cleanup

```bash
# Docker Compose
python engine_controller.py --action cleanup --mode docker

# Minikube
python engine_controller.py --action cleanup --mode local
# Or manually
minikube delete

# AWS (Helm only - cluster cleanup is manual)
python engine_controller.py --action cleanup --mode cloud_aws
# Then manually delete EKS cluster, RDS, ElastiCache, etc.
```

---

## Next Steps

1. **Learn the Python SDK**: `clients/python/README.md`
2. **Deploy a model**: Use the "Creating an Inference Endpoint" section above
3. **Fine-tune a model**: Explore fine-tuning examples in `examples/`
4. **Read the expert assessment**: `EXPERT_ASSESSMENT.md` for production considerations

---

## Support & Resources

- **Main Documentation**: https://scaleapi.github.io/llm-engine/
- **GitHub Issues**: https://github.com/scaleapi/llm-engine/issues
- **Architecture Deep Dive**: See `model-engine/README.md`
- **Self-Hosting Guide**: `docs/guides/self_hosting.md`

---

**Last Updated**: December 6, 2025
**Version**: 1.0

