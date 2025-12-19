# Model Engine Upgrade Plan - Queue Timeout Feature

## 📋 Current State Analysis

### Cluster Information
- **Cluster:** `sgpaz98031021k8s`
- **Resource Group:** `SGP98031021`
- **Namespace:** `launch`
- **Helm Release:** `launch-inference` (revision 1)
- **Chart Version:** `model-engine-0.1.13`

### Current Deployments
1. **model-engine** (Gateway)
   - Replicas: 2
   - Image: `022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:6e35c71cf82622fe2ad6e745728a65a1ff6f3984`
   - Health check: `/readyz` on port 5000

2. **model-engine-endpoint-builder**
   - Replicas: 1
   - Image: Same as gateway

3. **model-engine-cacher**
   - Replicas: 0 (scaled down)

### Key Configuration (From Helm Values)
```yaml
azure:
  abs_account_name: sgpaz98031021storage
  abs_container_name: sgpaz98031021models
  client_id: 1ef7e168-08e1-4798-b29b-7f5f9bd048ea
  servicebus_namespace: sgpaz98031021llm-engine
  keyvault_name: sgpaz98031021keyvault

config:
  values:
    infra:
      cloud_provider: azure
      docker_repo_prefix: 022465994601.dkr.ecr.us-west-2.amazonaws.com
      k8s_cluster_name: sgpaz98031021k8s
    launch:
      cache_redis_azure_host: sgpaz98031021rediscache.redis.cache.windows.net:6380
      endpoint_namespace: launch-inference
```

---

## 🎯 Upgrade Strategy: Rolling Update with New Image

### Why This Approach?
✅ **Preserves all existing configuration** (env vars, secrets, volumes)
✅ **Zero downtime** with rolling update
✅ **Easy rollback** via Helm or kubectl
✅ **Minimal risk** - only image changes

### What Will Change?
- ✅ Docker image tag (new image with your code)
- ❌ **NO** environment variables
- ❌ **NO** secrets or credentials
- ❌ **NO** ConfigMaps
- ❌ **NO** resource limits
- ❌ **NO** replica counts

---

## 📝 Step-by-Step Upgrade Process

### Phase 1: Build New Image
```bash
cd /home/20.scai.v.dragan/repos/llm-engine

# Build image with your code changes
docker build -t 022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:queue-timeout-$(date +%Y%m%d-%H%M%S) \
  -f model-engine/Dockerfile .
```

### Phase 2: Push to ECR
```bash
# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 022465994601.dkr.ecr.us-west-2.amazonaws.com

# Push image
docker push 022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:queue-timeout-TIMESTAMP
```

### Phase 3: Upgrade Helm Release
```bash
# Upgrade with new image tag ONLY
helm upgrade launch-inference /home/20.scai.v.dragan/repos/llm-engine/charts/model-engine \
  --namespace launch \
  --set tag=queue-timeout-TIMESTAMP \
  --reuse-values \
  --wait \
  --timeout 10m
```

**CRITICAL:** The `--reuse-values` flag ensures ALL existing values (including env vars, secrets, etc.) are preserved!

### Phase 4: Monitor Deployment
```bash
# Watch the rollout
kubectl rollout status deployment/model-engine -n launch -w

# In another terminal, watch pods
kubectl get pods -n launch -w | grep model-engine

# Check logs
kubectl logs -f deployment/model-engine -n launch
```

---

## 🛡️ Safety Measures

### ✅ Backups Created
All backups are in `/home/20.scai.v.dragan/repos/llm-engine/backups/`:
- `helm-values-backup.yaml` - Current Helm values
- `model-engine-deployment-backup.yaml` - Gateway deployment
- `endpoint-builder-deployment-backup.yaml` - Builder deployment
- `current-image.txt` - Current image tag
- `health-checks.txt` - Health check configuration
- `current-env-vars.txt` - Environment variables
- `quick-rollback.sh` - Executable rollback script

### ✅ Rollback Options

**Option 1: Helm Rollback (Recommended)**
```bash
cd /home/20.scai.v.dragan/repos/llm-engine/backups
./quick-rollback.sh
```

**Option 2: Manual kubectl**
```bash
kubectl set image deployment/model-engine \
  model-engine=022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:6e35c71cf82622fe2ad6e745728a65a1ff6f3984 \
  -n launch
```

### ✅ Rolling Update Configuration
Current strategy:
- `maxSurge: 25%` - Can create 1 extra pod during update (2 replicas * 0.25 = 0.5, rounded up to 1)
- `maxUnavailable: 0` - Always keep at least 2 pods running
- **Result:** Zero downtime guaranteed

---

## ⚠️ Potential Failure Scenarios & Mitigations

### 1. Image Pull Failure
**Symptoms:** Pods stuck in `ImagePullBackOff`

**Mitigation:**
- Test image pull before upgrade:
  ```bash
  docker pull 022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:queue-timeout-TIMESTAMP
  ```

**Rollback:** Automatic - old pods keep running

---

### 2. Health Check Failure
**Symptoms:** New pods never become Ready

**Mitigation:**
- Health check endpoint: `/readyz` on port 5000
- Timeout: 1 second
- Failure threshold: 30 attempts
- **Your code must respond to `/readyz` within 1 second**

**Rollback:** Run `./quick-rollback.sh`

---

### 3. Code Crashes on Startup
**Symptoms:** Pods crash loop with `CrashLoopBackOff`

**Mitigation:**
- Check logs immediately:
  ```bash
  kubectl logs -l app=model-engine -n launch --tail=100
  ```

**Rollback:** Run `./quick-rollback.sh`

---

### 4. Configuration Mismatch
**Symptoms:** Pods run but API errors

**Mitigation:**
- Your code changes should be **backward compatible**
- Don't change:
  - API endpoints
  - Environment variable names
  - Database schema
  - ConfigMap structure

**Rollback:** Run `./quick-rollback.sh`

---

## ✅ Pre-Upgrade Checklist

Before running the upgrade:

- [ ] All backups created (check `/backups/` directory)
- [ ] Rollback script tested (`./quick-rollback.sh --help`)
- [ ] Docker image built successfully
- [ ] Image pushed to ECR
- [ ] Image can be pulled from ECR
- [ ] Code changes are backward compatible
- [ ] Health check endpoint (`/readyz`) works in your code
- [ ] No database migrations required
- [ ] Team notified about upgrade

---

## 📊 Monitoring During Upgrade

### Terminal 1: Watch Rollout
```bash
kubectl rollout status deployment/model-engine -n launch -w
```

### Terminal 2: Watch Pods
```bash
kubectl get pods -n launch -w | grep model-engine
```

### Terminal 3: Watch Logs
```bash
kubectl logs -f deployment/model-engine -n launch
```

### Terminal 4: Watch Events
```bash
kubectl get events -n launch --sort-by='.lastTimestamp' -w | grep model-engine
```

---

## 🎉 Post-Upgrade Verification

After upgrade completes:

```bash
# 1. Check all pods are running
kubectl get pods -n launch | grep model-engine

# 2. Verify new image is deployed
kubectl get deployment model-engine -n launch -o jsonpath='{.spec.template.spec.containers[0].image}'

# 3. Check pod health
kubectl get pods -n launch -l app=model-engine -o wide

# 4. Test API health endpoint
kubectl port-forward -n launch svc/model-engine 8080:80
# In another terminal:
curl http://localhost:8080/health
curl http://localhost:8080/readyz

# 5. Test queue timeout feature
python /home/20.scai.v.dragan/repos/llm-engine/test_complete_queue_timeout_flow.py

# 6. Verify Azure Service Bus queues
az servicebus queue list \
  --namespace-name sgpaz98031021llm-engine \
  --resource-group SGP98031021 \
  --query "[].{name:name, lockDuration:lockDuration}" \
  --output table
```

---

## 🔄 Environment Variables - What's Preserved?

**The `--reuse-values` flag ensures these are NOT changed:**

From current deployment:
- Azure credentials (`client_id`, `identity_name`, etc.)
- Service Bus namespace
- Redis configuration
- Storage account settings
- All secrets and ConfigMaps
- Resource limits
- Replica counts
- Health check configuration

**Only the image tag changes!**

---

## 📞 Emergency Contacts

If something goes wrong:
1. **Immediate:** Run `./backups/quick-rollback.sh`
2. **Check logs:** `kubectl logs -l app=model-engine -n launch --tail=100`
3. **Check events:** `kubectl get events -n launch | grep model-engine`
4. **Escalate:** Contact team lead

---

## 🎯 Success Criteria

Upgrade is successful when:
- ✅ All pods are Running and Ready
- ✅ Health checks passing (`/readyz` returns 200)
- ✅ API endpoints responding
- ✅ Can create endpoints with `queue_message_timeout_duration`
- ✅ Azure Service Bus queues created with correct lock duration
- ✅ No error logs in pods
- ✅ Helm release shows as "deployed"

---

## Next Steps

Ready to proceed? Run:
```bash
cd /home/20.scai.v.dragan/repos/llm-engine
./build_and_deploy_queue_timeout.sh
```

This script will:
1. Build the Docker image with your changes
2. Push to ECR
3. Upgrade the Helm release
4. Monitor the deployment
5. Verify success
