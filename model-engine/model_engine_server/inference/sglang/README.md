# SGLang Multinode

## SGLang Startup Script

This script details what's necessary for each pod (leader and worker) to run to setup multinode SGLang. The most important details to note are:

1. `SGLANG_HOST_IP` is a critical environment variable needed to be set to the IP address of the leader pod of the k8s leaderworkerset (sglang source code [here](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils.py#L1362)).Ideally, this IP address would be DNS resolved (e.g. XXX.svc.cluster.local) but to avoid relying on an external Cluster service, we fetch the IP address manually once the leader pod is scheduled on a node in the startup script.

2. The `ml-serving-new` cluster runs on IPv6 pod addresses, so make sure the `host` for the SGLang server is `"::"` to indicate IPv6.

3. The `--port` argument to the SGLang server indicates the worker port, while the port followed by the `:` in the `--dist-init-addr` indicates the port that allows SGLang to initialize its underlying PyTorch distributed processes. These 2 ports MUST be different.

4. The worker port passed into the `--port` argument to the SGLang server must be `5005` to allow the Launch http forwarder to forward requests to the multinode SGLang server.

5. The script has logic that ensures the worker pod is scheduled before spinning up the server on the leader pod. This is necessary because for a leaderworkerset, pods will spin down / reschedule fairly quickly if the readinessprobe fails.

## Docker Image

`Dockerfile.sglang` details all of the necessary dependencies to spin up the multinode SGLang server. Current Launch endpoints (DeepSeek R1/V3) use a built/pushed docker image `sglang/multinode-latest-2`.

## Model Engine Code Changes
The code changes will only be in `model-engine/model_engine_server/domain/use_cases/llm_model_endpoint_use_cases.py`.

Update `CreateLLMModelEndpointV1UseCase` to accept SGLang multinode:
```python
if (
        request.nodes_per_worker > 1
        and not request.inference_framework in [LLMInferenceFramework.VLLM, LLMInferenceFramework.SGLANG]
    ):
        raise ObjectHasInvalidValueException(
            "Multinode endpoints are only supported for VLLM models."
        )
```

In `CreateLLMModelBundleV1UseCase`, add the following method:

```python
    async def create_sglang_multinode_bundle(
        self,
        user: User,
        model_name: str,
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
        nodes_per_worker: int,
        quantize: Optional[Quantization],
        checkpoint_path: Optional[str],
        chat_template_override: Optional[str],
        additional_args: Optional[SGLangEndpointAdditionalArgs] = None,
    ):
        leader_command = [
        "python3",
        "/root/sglang-startup-script.py",
        "--model",
        "deepseek-ai/DeepSeek-R1",
        "--nnodes",
        "2",
        "--node-rank",
        "0"
        "--worker-port",
        "5005",
        "--leader-port",
        "5002"
        ]
    
        worker_command = [
        "python3",
        "/root/sglang-startup-script.py",
        "--model",
        "deepseek-ai/DeepSeek-R1",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--worker-port",
        "5005",
        "--leader-port",
        "5002"
        ]

        # NOTE: the most important env var SGLANG_HOST_IP is already established in the sglang startup script
        
        common_sglang_envs = { # these are for debugging
            "NCCL_SOCKET_IFNAME": "eth0",
            "GLOO_SOCKET_IFNAME": "eth0",  
        }

        # This is same as VLLM multinode bundle
        create_model_bundle_v2_request = CreateModelBundleV2Request(
            name=endpoint_unique_name,
            schema_location="TBA",
            flavor=StreamingEnhancedRunnableImageFlavor(
                flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                repository=hmi_config.sglang_repository,
                tag=framework_image_tag,
                command=leader_command,
                streaming_command=leader_command,
                protocol="http",
                readiness_initial_delay_seconds=10,
                healthcheck_route="/health",
                predict_route="/predict",
                streaming_predict_route="/stream",
                extra_routes=[OPENAI_CHAT_COMPLETION_PATH, OPENAI_COMPLETION_PATH],
                env=common_sglang_envs,
                worker_command=worker_command,
                worker_env=common_sglang_envs,
            ),
            metadata={},
        )

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                create_model_bundle_v2_request,
                do_auth_check=False,
            )
        ).model_bundle_id
```


In `async def execute()`, now add SGLang as a multinode framework and call the sglang multinode bundle function under `case LLMInferenceFramework.SGLANG`:

```python
if multinode and framework not in [LLMInferenceFramework.VLLM, LLMInferenceFramework.SGLANG]:
    raise ObjectHasInvalidValueException(
        f"Multinode is not supported for framework {framework}."
    )

...

if multinode:
    bundle_id = await self.create_sglang_multinode_bundle(
        user,
        model_name,
        framework_image_tag,
        endpoint_name,
        num_shards,
        nodes_per_worker,
        quantize,
        checkpoint_path,
        chat_template_override,
        additional_args=additional_sglang_args,
    )
```
## Create multinode SGLang endpoint
1. Login to ECR:
```bash
aws ecr get-login-password --region us-west-2 | docker login \
  --username AWS \
  --password-stdin 692474966980.dkr.ecr.us-west-2.amazonaws.com
```
2. Push local `model-engine-internal` image for the forwarder:

```bash
docker push 692474966980.dkr.ecr.us-west-2.amazonaws.com/model-engine-internal:$(git rev-parse HEAD)-rcX
```

3. Spin up the local Launch server in `models/model-engine-internal` by running:
```bash
GIT_SHA="$(git rev-parse HEAD)-rcX" SETENV=prod LOCAL=true AWS_PROFILE=ml-serving-admin just up prod
```

4. Send a `POST` request to `http://localhost:5001/v1/llm/model-endpoints` with the following body:

```json
{
  "name": "deepseek-r1",
  "model_name": "deepseek-r1",
  "endpoint_type": "streaming",
  "cpus": 160,
  "memory": "800Gi",
  "min_workers": 1,
  "max_workers": 1,
  "gpus": 8,
  "gpu_type": "nvidia-hopper-h100",
  "storage": "900Gi",
  "per_worker": 1,
  "num_shards": 8,
  "nodes_per_worker": 2,
  "labels": {
    "team": "infra",
    "product": "inference.llm_model_zoo"
  },
  "inference_framework": "sglang",
  "inference_framework_image_tag": "multinode-latest-2",
  "high_priority": true,
  "metadata": {
      "_llm": 
        {"source": "hugging_face", "quantize": null, "model_name": "deepseek-r1", "num_shards": 8, "checkpoint_path": "s3://scale-ml/models/hf-synced-weights/deepseek-ai/DeepSeek-R1", "inference_framework": "sglang", "chat_template_override": null, "inference_framework_image_tag": "multinode-latest-2"}
  }
}
```
