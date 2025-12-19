# Rollback Instructions

**Backup Created:** $(cat backup-timestamp.txt)
**Current Image:** `022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:6e35c71cf82622fe2ad6e745728a65a1ff6f3984`
**Helm Release:** `launch-inference`
**Namespace:** `launch`
**Current Revision:** 1

---

## 🚨 Quick Rollback (Recommended)

If the upgrade fails, use Helm rollback:

```bash
# Rollback to previous revision (revision 1)
helm rollback launch-inference 1 -n launch

# Monitor the rollback
kubectl rollout status deployment/model-engine -n launch
kubectl rollout status deployment/model-engine-endpoint-builder -n launch

# Verify pods are healthy
kubectl get pods -n launch | grep model-engine
```

---

## 🔧 Manual Rollback (If Helm Fails)

If Helm rollback doesn't work, manually set the image:

```bash
# Rollback model-engine deployment
kubectl set image deployment/model-engine \
  model-engine=022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:6e35c71cf82622fe2ad6e745728a65a1ff6f3984 \
  -n launch

# Rollback endpoint-builder deployment
kubectl set image deployment/model-engine-endpoint-builder \
  model-engine-endpoint-builder=022465994601.dkr.ecr.us-west-2.amazonaws.com/egp-mirror-int/model-engine:6e35c71cf82622fe2ad6e745728a65a1ff6f3984 \
  -n launch

# Monitor rollback
kubectl rollout status deployment/model-engine -n launch
kubectl rollout status deployment/model-engine-endpoint-builder -n launch
```

---

## 🆘 Emergency Restore (Nuclear Option)

If everything fails, restore from backup files:

```bash
cd /home/20.scai.v.dragan/repos/llm-engine/backups

# Delete current deployments
kubectl delete deployment model-engine -n launch
kubectl delete deployment model-engine-endpoint-builder -n launch

# Wait for deletion
sleep 10

# Restore from backup
kubectl apply -f model-engine-deployment-backup.yaml
kubectl apply -f endpoint-builder-deployment-backup.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=model-engine -n launch --timeout=300s
```

---

## ✅ Verification After Rollback

```bash
# Check pods are running
kubectl get pods -n launch | grep model-engine

# Check image is correct
kubectl get deployment model-engine -n launch -o jsonpath='{.spec.template.spec.containers[0].image}'

# Check logs for errors
kubectl logs -l app=model-engine -n launch --tail=50

# Test the API
kubectl port-forward -n launch svc/model-engine 8080:80
# Then in another terminal: curl http://localhost:8080/health
```

---

## 📊 Monitoring Commands

```bash
# Watch pods during rollback
kubectl get pods -n launch -w | grep model-engine

# Check events
kubectl get events -n launch --sort-by='.lastTimestamp' | grep model-engine

# View logs
kubectl logs -f deployment/model-engine -n launch
```
