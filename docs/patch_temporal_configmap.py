#!/usr/bin/env python3
"""
Patch the model-engine-service-template-config ConfigMap on ml-serving-new to add
temporal CPU and GPU deployment templates.

Usage:
    python3 docs/patch_temporal_configmap.py

Requires: kubectl context pointing at ml-serving-new.
"""

import json
import subprocess

TEMPORAL_CPU_TEMPLATE = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${RESOURCE_NAME}
  namespace: ${NAMESPACE}
  labels:
    user_id: ${OWNER}
    team: ${TEAM}
    product: ${PRODUCT}
    created_by: ${CREATED_BY}
    owner: ${OWNER}
    env: prod
    managed-by: model-engine
    use_scale_launch_endpoint_network_policy: "true"
    tags.datadoghq.com/env: prod
    tags.datadoghq.com/version: ${GIT_TAG}
    tags.datadoghq.com/service: ${ENDPOINT_NAME}
    endpoint_id: ${ENDPOINT_ID}
    endpoint_name: ${ENDPOINT_NAME}
  annotations:
    temporal.scaleml.io/taskQueue: "${TEMPORAL_TASK_QUEUE}"
    temporal.scaleml.io/minWorkers: "${MIN_WORKERS}"
    temporal.scaleml.io/maxWorkers: "${MAX_WORKERS}"
    temporal.scaleml.io/perWorker: "${PER_WORKER}"
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  replicas: ${MAX_WORKERS}
  selector:
    matchLabels:
      app: ${RESOURCE_NAME}
      version: v1
  template:
    metadata:
      labels:
        app: ${RESOURCE_NAME}
        user_id: ${OWNER}
        team: ${TEAM}
        product: ${PRODUCT}
        created_by: ${CREATED_BY}
        owner: ${OWNER}
        env: prod
        managed-by: model-engine
        use_scale_launch_endpoint_network_policy: "true"
        tags.datadoghq.com/env: prod
        tags.datadoghq.com/version: ${GIT_TAG}
        tags.datadoghq.com/service: ${ENDPOINT_NAME}
        endpoint_id: ${ENDPOINT_ID}
        endpoint_name: ${ENDPOINT_NAME}
        sidecar.istio.io/inject: "false"
        version: v1
      annotations:
        ad.datadoghq.com/main.logs: '[{"service": "${ENDPOINT_NAME}", "source": "python"}]'
        kubernetes.io/change-cause: "${CHANGE_CAUSE_MESSAGE}"
    spec:
      affinity:
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 1
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ${RESOURCE_NAME}
              topologyKey: kubernetes.io/hostname
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: ${IMAGE_HASH}
                  operator: In
                  values:
                  - "True"
              topologyKey: kubernetes.io/hostname
      terminationGracePeriodSeconds: 1800
      serviceAccount: ml-worker
      nodeSelector:
        node-lifecycle: normal
      priorityClassName: ${PRIORITY}
      containers:
        - name: main
          image: ${IMAGE}
          imagePullPolicy: IfNotPresent
          command: ${COMMAND}
          env: ${MAIN_ENV}
          resources:
            requests:
              cpu: ${CPUS}
              memory: ${MEMORY}
              ${STORAGE_DICT}
            limits:
              cpu: ${CPUS}
              memory: ${MEMORY}
              ${STORAGE_DICT}
          volumeMounts:
            - name: user-config
              mountPath: /app/user_config
              subPath: raw_data
            - name: endpoint-config
              mountPath: /app/endpoint_config
              subPath: raw_data
            - name: infra-service-config-volume
              mountPath: ${INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH}
      securityContext:
        fsGroup: 65534
      volumes:
        - name: user-config
          configMap:
            name: ${RESOURCE_NAME}
        - name: endpoint-config
          configMap:
            name: ${RESOURCE_NAME}-endpoint-config
        - name: infra-service-config-volume
          configMap:
            name: model-engine-service-config
            items:
              - key: infra_service_config
                path: config.yaml"""

TEMPORAL_GPU_TEMPLATE = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${RESOURCE_NAME}
  namespace: ${NAMESPACE}
  labels:
    user_id: ${OWNER}
    team: ${TEAM}
    product: ${PRODUCT}
    created_by: ${CREATED_BY}
    owner: ${OWNER}
    env: prod
    managed-by: model-engine
    use_scale_launch_endpoint_network_policy: "true"
    tags.datadoghq.com/env: prod
    tags.datadoghq.com/version: ${GIT_TAG}
    tags.datadoghq.com/service: ${ENDPOINT_NAME}
    endpoint_id: ${ENDPOINT_ID}
    endpoint_name: ${ENDPOINT_NAME}
  annotations:
    temporal.scaleml.io/taskQueue: "${TEMPORAL_TASK_QUEUE}"
    temporal.scaleml.io/minWorkers: "${MIN_WORKERS}"
    temporal.scaleml.io/maxWorkers: "${MAX_WORKERS}"
    temporal.scaleml.io/perWorker: "${PER_WORKER}"
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  replicas: ${MAX_WORKERS}
  selector:
    matchLabels:
      app: ${RESOURCE_NAME}
      version: v1
  template:
    metadata:
      labels:
        app: ${RESOURCE_NAME}
        user_id: ${OWNER}
        team: ${TEAM}
        product: ${PRODUCT}
        created_by: ${CREATED_BY}
        owner: ${OWNER}
        env: prod
        managed-by: model-engine
        use_scale_launch_endpoint_network_policy: "true"
        tags.datadoghq.com/env: prod
        tags.datadoghq.com/version: ${GIT_TAG}
        tags.datadoghq.com/service: ${ENDPOINT_NAME}
        endpoint_id: ${ENDPOINT_ID}
        endpoint_name: ${ENDPOINT_NAME}
        sidecar.istio.io/inject: "false"
        version: v1
      annotations:
        ad.datadoghq.com/main.logs: '[{"service": "${ENDPOINT_NAME}", "source": "python"}]'
        kubernetes.io/change-cause: "${CHANGE_CAUSE_MESSAGE}"
    spec:
      affinity:
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 1
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ${RESOURCE_NAME}
              topologyKey: kubernetes.io/hostname
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: ${IMAGE_HASH}
                  operator: In
                  values:
                  - "True"
              topologyKey: kubernetes.io/hostname
      terminationGracePeriodSeconds: 1800
      serviceAccount: ml-worker
      nodeSelector:
        node-lifecycle: normal
        k8s.amazonaws.com/accelerator: ${GPU_TYPE}
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      priorityClassName: ${PRIORITY}
      containers:
        - name: main
          image: ${IMAGE}
          imagePullPolicy: IfNotPresent
          command: ${COMMAND}
          env: ${MAIN_ENV}
          resources:
            requests:
              nvidia.com/gpu: ${GPUS}
              cpu: ${CPUS}
              memory: ${MEMORY}
              ${STORAGE_DICT}
            limits:
              nvidia.com/gpu: ${GPUS}
              cpu: ${CPUS}
              memory: ${MEMORY}
              ${STORAGE_DICT}
          volumeMounts:
            - mountPath: /dev/shm
              name: dshm
            - name: user-config
              mountPath: /app/user_config
              subPath: raw_data
            - name: endpoint-config
              mountPath: /app/endpoint_config
              subPath: raw_data
            - name: infra-service-config-volume
              mountPath: ${INFRA_SERVICE_CONFIG_VOLUME_MOUNT_PATH}
      securityContext:
        fsGroup: 65534
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: user-config
          configMap:
            name: ${RESOURCE_NAME}
        - name: endpoint-config
          configMap:
            name: ${RESOURCE_NAME}-endpoint-config
        - name: infra-service-config-volume
          configMap:
            name: model-engine-service-config
            items:
              - key: infra_service_config
                path: config.yaml"""


def main():
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "configmap",
            "model-engine-service-template-config",
            "-n",
            "default",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cm = json.loads(result.stdout)
    cm["data"]["deployment-runnable-image-temporal-cpu.yaml"] = TEMPORAL_CPU_TEMPLATE
    cm["data"]["deployment-runnable-image-temporal-gpu.yaml"] = TEMPORAL_GPU_TEMPLATE

    result = subprocess.run(
        ["kubectl", "apply", "-f", "-", "-n", "default"],
        input=json.dumps(cm),
        capture_output=True,
        text=True,
    )
    print(result.stdout or result.stderr)

    keys = [k for k in cm["data"] if "temporal" in k]
    print(f"Temporal template keys added: {keys}")


if __name__ == "__main__":
    main()
