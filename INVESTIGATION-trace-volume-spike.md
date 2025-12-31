# Datadog Trace Volume Spike Investigation

**Date**: December 30-31, 2025
**Cluster**: ml-serving-new
**Metric**: `sum:trace.envoy.proxy.hits.by_http_status{$cluster_name AND (resource_name:launch* OR resource_name:model-engine*) AND NOT (service:model-engine* OR service:launch*)} by {http.status_code}.as_count()`

## Summary

High trace volume (~62k hits per 10 seconds) was observed starting around December 27, 2025. Investigation revealed the spike was caused by:

1. **Datadog agent restart** on Dec 25-26 fixing broken trace collection
2. **High-volume async task polling** from a single team running ~2500 concurrent tasks

## Timeline

| Date | Event |
|------|-------|
| Dec 25-26 | Datadog helm upgrade (chart 3.74.1 → 3.116.1, agent 7.57.2 → 7.65.2) |
| Dec 25-26 | Cluster agent token rotation triggered rolling restart of all datadog agents |
| Dec 27 | Trace volume spike observed in Datadog |

## Root Cause

### Primary Source: Async Task Polling

**Metric identified**: `istio.mesh.request.count.total` with `request_operation=/v1/async-tasks/_task_id`

**Team**: `63bf16bdc65a8abb84ad8615`
**Endpoint**: `archie-copilot-async`

**Traffic pattern**:
- ~2500 concurrent async tasks at any given time
- Each task polled every ~3 seconds
- = ~830 polls/second = ~50,000 polls/minute
- Each poll generates a trace

**Deployments serving this workload**:
| Deployment | Pods | Age |
|------------|------|-----|
| `launch-endpoint-id-end-cv7khs1qcvdg031ru3m0` | 10/10 | 295d |
| `launch-endpoint-id-end-d3ohvace85eg02k1pbq0` | 4/4 | 75d |

### Secondary Factor: Istio Trace Sampling

Istio's tracing configuration has **no sampling rate configured**, defaulting to 100% trace sampling. This means every request generates a trace.

```yaml
# From istio-configmap (istio-system/istio)
meshConfig:
  defaultConfig:
    tracing:
      zipkin:
        address: [$(HOST_IP)]:8126
      # No sampling rate configured = 100% sampling
```

### Tertiary Factor: Retry Logic

model-engine's sync endpoint gateway (`live_sync_model_endpoint_inference_gateway.py`) has aggressive retry logic:

```python
SYNC_ENDPOINT_RETRIES = 8  # Up to 9 total attempts per request
```

When endpoints fail (503/429), each retry generates a new trace, multiplying trace volume.

## Why Traces Weren't Visible Before

The datadog agent upgrade on Dec 25-26 likely fixed broken trace collection. The traffic pattern (async polling) existed before, but traces weren't being collected/forwarded properly.

## Recommendations

### Short-term (No Code Changes)

1. **Configure Istio trace sampling** to 1-5%
   - Edit `istio` configmap in `istio-system` namespace
   - Add `sampling: 1.0` (for 1%) under tracing config

### Medium-term (Code Changes)

2. **Increase client poll interval** from 3s to 10-30s for async tasks
   - Coordinate with team `63bf16bdc65a8abb84ad8615`
   - Update client SDK defaults

3. **Reduce retry count** from 8 to 2-3
   - File: `model_engine_server/infra/gateways/live_sync_model_endpoint_inference_gateway.py`
   - Line 40: `SYNC_ENDPOINT_RETRIES = 8` → `SYNC_ENDPOINT_RETRIES = 3`

### Long-term (Architecture)

4. **Implement webhooks/callbacks** for async task completion
   - Eliminates polling entirely
   - Server pushes completion status to client

5. **Add per-team rate limiting** on `/v1/async-tasks/{task_id}` endpoint

## Investigation Commands Used

```bash
# Check trace agent logs
kubectl --context ml-serving-new logs -n datadog -l app=datadog -c trace-agent --tail=100

# Count async-task polls
kubectl --context ml-serving-new logs -n default -l app=model-engine --tail=500 --since=10s | grep "GET /async-tasks" | wc -l

# Identify top polling teams
kubectl --context ml-serving-new logs -n default -l app=model-engine --tail=3000 --since=30s | grep "GET /async-tasks" | grep -o '"team_id": "[^"]*"' | sort | uniq -c | sort -rn

# Count unique concurrent tasks
kubectl --context ml-serving-new logs -n default -l app=model-engine --tail=500 --since=10s | grep "GET /async-tasks" | grep -o 'async-tasks/[a-f0-9-]*' | sort -u | wc -l
```

## Files Referenced

- `model-engine/model_engine_server/infra/gateways/live_sync_model_endpoint_inference_gateway.py` - Retry logic
- `model-engine/model_engine_server/api/tasks_v1.py` - Async tasks API endpoint
- Istio configmap: `istio-system/istio` - Tracing configuration

## Proposed Solution

### Trace Sampling via EnvoyFilter (Terracode)

To reduce trace volume, add an EnvoyFilter in Terracode (`scaleapi-ml-serving/clusters/ml-serving-new/istio.tf`) that applies to the **ingress gateway** (not model-engine sidecar, since traces originate at ingress).

**Key considerations:**
1. EnvoyFilter must target `istio-ingressgateway-internal` with `context: GATEWAY`
2. Lua filter may not be available in default gateway image - verify or use alternative approach
3. Need to set both `x-b3-sampled: 0` and `x-datadog-sampling-priority: 0` headers

**Alternative approaches if Lua not available:**
1. Configure Datadog agent's `apm_config.ignore_resources` to drop traces matching `/v1/async-tasks/*`
2. Use Datadog's ingestion controls to sample at collection time
3. Add trace sampling configuration to Istio mesh config

**Expected impact:**
- Reduce trace volume from ~50k/min to ~500/min for async-tasks polling
- ~99% reduction in trace costs for this endpoint
