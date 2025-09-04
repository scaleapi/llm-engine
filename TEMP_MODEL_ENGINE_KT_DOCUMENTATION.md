# Model Engine Knowledge Transfer Documentation

## 🎯 Executive Summary & Current State

### **Current Working Configuration (STABLE)**
- **Model-Engine Image**: `onprem20` (PVC removed, optimized AWS CLI)
- **VLLM Image**: `vllm-onprem` (model architecture fixes for Qwen3ForCausalLM)
- **Storage Configuration**: `50GB ephemeral` (prevents container termination)
- **Status**: First stable endpoint deployment with active model downloads (8+ minutes uptime, 655MB downloaded)

---

## **Image Ecosystem & Build Process**

#### **Model-Engine Images** (Business Logic)
- **Repository**: `registry.odp.om/odp-development/oman-national-llm/model-engine:onpremXX`
- **Contents**: Python application code, endpoint builder logic, Kubernetes resource generation
- **Build Source**: `llm-engine` repository
- **Current Working**: `onprem20` (PVC removed, optimized AWS CLI)
- **Build Trigger**: Code changes in llm-engine repository

#### **VLLM Images** (Inference Runtime)  
- **Repository**: `registry.odp.om/odp-development/oman-national-llm/vllm:TAG`
- **Contents**: VLLM inference framework, model serving logic, runtime dependencies
- **Build Source**: Separate VLLM Dockerfile (not in main repos)
- **Current Working**: `vllm-onprem` (with Qwen3ForCausalLM compatibility)
- **Build Trigger**: VLLM framework updates or model architecture fixes

#### **Image Relationship**
```
model-engine image (onprem20) 
    ↓ (generates Kubernetes manifests)
VLLM container (vllm-onprem)
    ↓ (downloads models and runs inference)
Model Files (S3) → VLLM Server → API Endpoints
```

### **Storage Architecture**
- **Ephemeral Storage**: Node-local, lost on pod restart, 189GB total capacity
- **PVC Storage**: Persistent, Ceph RBD backed, attempted but has async bugs
- **Current**: Using ephemeral with 50GB limits (within node capacity)

---





#### **Model Architecture Compatibility**
- **Problem**: `ValueError: Model architectures ['Qwen3ForCausalLM'] are not supported`
- **Impact**: VLLM failed to load Qwen3 models
- **Solution**: Updated to `vllm-onprem` image with architecture fixes

---

## 🔧 Technical Deep Dive

### **Working Configuration Details**

#### **Image Configuration**
```yaml
# values.yaml
tag: onprem20
vllm_repository: "odp-development/oman-national-llm/vllm"
vllm_tag: "vllm-onprem"
```





### **S3 Integration Details**

#### **Working Environment Variables**
```bash
AWS_ACCESS_KEY_ID=<from-kubernetes-secret>
AWS_SECRET_ACCESS_KEY=<from-kubernetes-secret>
AWS_ENDPOINT_URL=https://oss.odp.om
AWS_REGION=us-east-1
AWS_EC2_METADATA_DISABLED=true
```

#### **S3 Download Command**
```bash
# Full command with environment variables
AWS_ACCESS_KEY_ID=<from-kubernetes-secret> \
AWS_SECRET_ACCESS_KEY=<from-kubernetes-secret> \
AWS_ENDPOINT_URL=https://oss.odp.om \
AWS_REGION=us-east-1 \
AWS_EC2_METADATA_DISABLED=true \
aws s3 sync s3://scale-gp-models/intermediate-model-aws model_files --no-progress

# S3 Endpoint Details
# Scality S3 Endpoint: https://oss.odp.om
# Bucket: scale-gp-models
# Path: intermediate-model-aws/
```

### **Timing Coordination Logic**
The working timing coordination waits for:
1. **config.json** file to exist
2. **All .safetensors files** to be present
3. **No temp suffixes** on any files (indicating AWS CLI completion)

### **Endpoint Creation Workflow**

When an endpoint is created via API call, here's the complete workflow:

#### **Step 1: API Request Processing**
```
curl -X POST /v1/llm/model-endpoints → model-engine service
```
- **model-engine** receives API request
- Validates parameters and creates endpoint record
- Queues build task for **endpoint-builder**

#### **Step 2: Kubernetes Resource Generation**
```
endpoint-builder → reads hardware config → generates K8s manifests
```
- **endpoint-builder** processes the build task
- Reads `recommendedHardware` from ConfigMap
- Generates template variables: `${STORAGE_DICT}`, `${WORKDIR_VOLUME_CONFIG}`
- Substitutes variables into deployment template
- Creates: Deployment, Service, HPA

#### **Step 3: Pod Scheduling & Container Creation**
```
K8s Scheduler → GPU Node → Container Creation
```
- **Scheduler** assigns pod to `hpc-k8s-phy-wrk-g01` (only GPU node)
- **kubelet** pulls images: `model-engine:onprem20`, `vllm:vllm-onprem`
- Creates **2 containers**: `http-forwarder` + `main`

#### **Step 4: Model Download & Preparation**
```
main container → AWS CLI install → S3 download → File verification
```
- **AWS CLI installation**: `pip install --quiet awscli --no-cache-dir`
- **S3 download**: `aws s3 sync s3://scale-gp-models/intermediate-model-aws model_files`
- **File verification**: Wait for temp suffixes to be removed
- **Timing coordination**: Verify `config.json` and `.safetensors` files ready

#### **Step 5: VLLM Server Startup**
```
Model files ready → VLLM startup → Health checks → Service ready
```
- **VLLM startup**: `python -m vllm_server --model model_files`
- **Health checks**: `/health` endpoint on port 5005
- **Service routing**: `http-forwarder` routes traffic to VLLM
- **Pod status**: Transitions from `0/2` → `2/2` Running

#### **Step 6: Inference Ready**
```
2/2 Running → Load balancer → External access
```
- Both containers healthy and ready
- Service endpoints accessible
- Ready for inference requests


### **Container Architecture**
```
Pod: launch-endpoint-id-end-{ID}
├── Container: http-forwarder (model-engine:onprem20)
│   └── Routes traffic to main container
└── Container: main (vllm:vllm-onprem)
    ├── AWS CLI installation (~5-10 min)
    ├── S3 model download (~30-60 min)
    ├── File verification & timing coordination
    └── VLLM server startup
```

---

## 🛠️ Operational Procedures

### **Testing Workflow**

#### **1. Deploy New Image Version**
```bash
# Update values.yaml tag, then:
kubectl rollout restart deployment model-engine -n llm-core
kubectl rollout restart deployment model-engine-endpoint-builder -n llm-core

# Verify image deployment
kubectl describe pod $(kubectl get pods -n llm-core | grep "model-engine" | head -1 | awk '{print $1}') -n llm-core | grep "Image:"
```

#### **2. Create Test Endpoint**
```bash
# Start port-forward
kubectl port-forward svc/model-engine -n llm-core 5000:80 &

# Create endpoint (50GB storage is critical!)
curl -X POST -H "Content-Type: application/json" -u "test-user-id:" "http://localhost:5000/v1/llm/model-endpoints" -d '{
  "name": "test-endpoint-v1",
  "model_name": "test-model",
  "endpoint_type": "streaming",
  "inference_framework": "vllm",
  "inference_framework_image_tag": "vllm-onprem",
  "source": "hugging_face",
  "checkpoint_path": "s3://scale-gp-models/intermediate-model-aws/",
  "num_shards": 1,
  "cpus": 4,
  "memory": "16Gi",
  "storage": "50Gi",
  "gpus": 1,
  "gpu_type": "nvidia-tesla-t4",
  "nodes_per_worker": 1,
  "min_workers": 1,
  "max_workers": 1,
  "per_worker": 1,
  "metadata": {"team": "test", "product": "llm-engine"},
  "labels": {"team": "test", "product": "llm-engine"}
}'
```

#### **3. Monitor Endpoint Progress**
```bash
# Check pod creation
kubectl get all -n llm-core | grep "launch-endpoint"

# Monitor container processes
kubectl exec ENDPOINT_POD -n llm-core -c main -- ps aux

# Check download progress
kubectl exec ENDPOINT_POD -n llm-core -c main -- ls -la model_files/
kubectl exec ENDPOINT_POD -n llm-core -c main -- du -sh model_files/

# Monitor logs
kubectl logs ENDPOINT_POD -n llm-core -c main --tail=10 -f
```

#### **4. Cleanup Failed Endpoints**
```bash
# Delete endpoint resources
kubectl delete deployment ENDPOINT_DEPLOYMENT -n llm-core
kubectl delete service ENDPOINT_SERVICE -n llm-core
kubectl delete hpa ENDPOINT_HPA -n llm-core

# Clean up old replica sets
kubectl get replicasets -n llm-core | grep model-engine | awk '$3 == 0 {print $1}' | xargs -r kubectl delete replicaset -n llm-core
```

### **Common Issues & Quick Fixes**

| Issue | Symptoms | Root Cause | Solution |
|-------|----------|------------|----------|
| **Container Termination** | Exit Code 137, pod dies in <5min | Storage limits exceeded | Use 50GB storage (not 100GB+) |
| **Slow AWS CLI Install** | 30+ minute installations | Missing optimization flag | Verify `--no-cache-dir` in command |
| **Architecture Errors** | `Qwen3ForCausalLM not supported` | Wrong VLLM image | Use `vllm-onprem` tag |
| **Download Fails** | No model_files directory | AWS CLI or S3 auth issues | Check `which aws`, verify credentials |
| **Premature VLLM Start** | `No config format found` | Timing coordination missing | Verify `while` loop in command |

### **Key Monitoring Commands**
```bash
# Check cluster storage capacity
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu',EPHEMERAL-STORAGE:.status.allocatable.ephemeral-storage

# Monitor active downloads
kubectl exec ENDPOINT_POD -n llm-core -c main -- ps aux | grep aws

# Check file finalization status
kubectl exec ENDPOINT_POD -n llm-core -c main -- ls -la model_files/ | grep -E "\.tmp|\..*[A-Za-z0-9]{8}$"

# Monitor endpoint builder
kubectl logs deployment/model-engine-endpoint-builder -n llm-core --tail=20
```

---

## 🚨 Known Issues & Future Work

### **Critical Unresolved Issues**

#### **1. PVC Functionality Broken**
- **Status**: All attempts to use PVC storage fail
- **Root Cause**: Async hardware config bug in appcode
- **Error**: `RuntimeWarning: coroutine '_get_recommended_hardware_config_map' was never awaited`
- **Impact**: Always falls back to EmptyDir instead of PVC
- **Workaround**: Using ephemeral storage with reduced limits
- **PVC Code Status**: PVC implementation has been **reverted from both repositories** and is **scheduled for rework next week**
- **Fix Required**: Changes to `llm-engine` repository to properly await async hardware config function

#### **2. Storage Scaling Limitations**
- **Current**: Single GPU node with 189GB ephemeral storage
- **Constraint**: Large models require more storage than available
- **Options**: Add GPU nodes, expand node storage, or implement working PVC

#### **3. Download Performance**
- **Current**: ~4MB/s download speeds from Scality S3
- **Optimization**: Could pre-install AWS CLI in base images
- **Alternative**: Use faster download tools or local mirrors

### **Prevention Guidelines**
- **Always use 50GB storage** for tesla-t4 hardware (not 100GB+)
- **Always use `vllm-onprem` tag** (not version-specific like `0.6.3-rc1`)
- **Always include `--no-cache-dir`** in AWS CLI installation commands
- **Test endpoint creation** immediately after any image updates
- **Monitor container uptime** - quick termination indicates problems

---

## 📁 Critical File Locations

### **oman-national-llm Repository**
```
infra/charts/model-engine/
├── values.yaml                                    # Main configuration
├── templates/
│   ├── service_template_config_map.yaml          # Pod/deployment templates
│   ├── recommended_hardware_config_map.yaml      # Hardware specifications
│   ├── service_config_map.yaml                   # Service configuration
│   └── _helpers.tpl                              # Helm helper functions
```



---

## 🚀 Quick Reference

### **Working API Call**
```bash
curl -X POST -H "Content-Type: application/json" -u "test-user-id:" "http://localhost:5000/v1/llm/model-endpoints" -d '{
  "name": "test-endpoint-v1",
  "model_name": "test-model", 
  "endpoint_type": "streaming",
  "inference_framework": "vllm",
  "inference_framework_image_tag": "vllm-onprem",
  "source": "hugging_face",
  "checkpoint_path": "s3://scale-gp-models/intermediate-model-aws/",
  "num_shards": 1,
  "cpus": 4,
  "memory": "16Gi",
  "storage": "50Gi",  # CRITICAL: Must be 50Gi or less
  "gpus": 1,
  "gpu_type": "nvidia-tesla-t4",
  "nodes_per_worker": 1,
  "min_workers": 1,
  "max_workers": 1,
  "per_worker": 1,
  "metadata": {"team": "test", "product": "llm-engine"},
  "labels": {"team": "test", "product": "llm-engine"}
}'
```

### **Emergency Revert Procedure**
```bash
# Revert to last working state
kubectl set image deployment/model-engine model-engine=registry.odp.om/odp-development/oman-national-llm/model-engine:onprem20 -n llm-core
kubectl set image deployment/model-engine-endpoint-builder model-engine-endpoint-builder=registry.odp.om/odp-development/oman-national-llm/model-engine:onprem20 -n llm-core

# Update values.yaml
tag: onprem20
vllm_tag: "vllm-onprem"

# Verify storage configuration
storage: 50Gi  # In hardware specs
```



---

*This documentation represents the culmination of extensive testing and debugging to achieve the first stable model-engine deployment. Preserve this configuration as the baseline for future development.*
