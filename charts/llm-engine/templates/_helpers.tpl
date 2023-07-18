{{/*
Expand the name of the chart.
*/}}
{{- define "llmEngine.name" -}}
{{- default .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 40 chars because some Kubernetes name fields are limited to 63 (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "llmEngine.fullname" -}}
{{- if .Values.serviceIdentifier }}
{{- printf "%s-%s" .Chart.Name .Values.serviceIdentifier | trunc 40 | trimSuffix "-" }}
{{- else }}
{{- default .Chart.Name | trunc 40 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "llmEngine.buildername" -}}
"{{ include "llmEngine.fullname" . }}-endpoint-builder"
{{- end }}

{{- define "llmEngine.cachername" -}}
"{{ include "llmEngine.fullname" . }}-cacher"
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "llmEngine.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "llmEngine.labels" -}}
team: infra
product: llm-engine
helm.sh/chart: {{ include "llmEngine.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Values.tag }}
tags.datadoghq.com/version: {{ .Values.tag }}
tags.datadoghq.com/env: {{ .Values.context }}
{{- end }}

{{- define "llmEngine.selectorLabels.builder" -}}
app: {{ include "llmEngine.buildername" . }}
{{- end }}

{{- define "llmEngine.selectorLabels.cacher" -}}
app: {{ include "llmEngine.cachername" . }}
{{- end }}

{{- define "llmEngine.selectorLabels.gateway" -}}
app: {{ include "llmEngine.fullname" . -}}
{{- end }}

{{- define "llmEngine.baseTemplateLabels" -}}
user_id: ${OWNER}
team: ${TEAM}
product: ${PRODUCT}
created_by: ${CREATED_BY}
owner: ${OWNER}
env: {{- .Values.context | printf " %s" }}
managed-by: {{- include "llmEngine.fullname" . | printf " %s\n" -}}
use_scale_llm_engine_endpoint_network_policy: "true"
tags.datadoghq.com/env: {{- .Values.context | printf " %s" }}
tags.datadoghq.com/version: {{- .Values.tag | printf " %s" }}
{{- end }}

{{- define "llmEngine.serviceTemplateLabels" -}}
{{- include "llmEngine.baseTemplateLabels" . | printf "%s\n" -}}
tags.datadoghq.com/service: ${ENDPOINT_NAME}
endpoint_id: ${ENDPOINT_ID}
endpoint_name: ${ENDPOINT_NAME}
{{- end }}

{{- define "llmEngine.jobTemplateLabels" -}}
{{- include "llmEngine.baseTemplateLabels" . | printf "%s\n" -}}
llm_engine_job_id: ${JOB_ID}
tags.datadoghq.com/service: ${JOB_ID}
{{- end }}

{{- define "llmEngine.serviceTemplateAsyncAnnotations" -}}
celery.scaleml.autoscaler/queue: ${QUEUE}
celery.scaleml.autoscaler/broker: ${BROKER_NAME}
celery.scaleml.autoscaler/taskVisibility: "VISIBILITY_24H"
celery.scaleml.autoscaler/perWorker: "${PER_WORKER}"
celery.scaleml.autoscaler/minWorkers: "${MIN_WORKERS}"
celery.scaleml.autoscaler/maxWorkers: "${MAX_WORKERS}"
{{- end }}

{{- define "llmEngine.serviceTemplateAffinity" -}}
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
{{- end }}

{{- define "llmEngine.baseServiceTemplateEnv" -}}
env:
  - name: DATADOG_TRACE_ENABLED
    value: "${DATADOG_TRACE_ENABLED}"
  - name: DD_SERVICE
    value: "${ENDPOINT_NAME}"
  - name: DD_ENV
    value: {{ .Values.context }}
  - name: DD_VERSION
    value: {{ .Values.tag }}
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: OMP_NUM_THREADS
    value: "1"
  - name: BASE_PATH
    value: "${BASE_PATH}"
  - name: BUNDLE_URL
    value: "${BUNDLE_URL}"
  - name: LOAD_PREDICT_FN_MODULE_PATH
    value: "${LOAD_PREDICT_FN_MODULE_PATH}"
  - name: LOAD_MODEL_FN_MODULE_PATH
    value: "${LOAD_MODEL_FN_MODULE_PATH}"
  - name: AWS_PROFILE
    value: "${AWS_ROLE}"
  - name: RESULTS_S3_BUCKET
    value: "${RESULTS_S3_BUCKET}"
  - name: CHILD_FN_INFO
    value: "${CHILD_FN_INFO}"
  - name: PREWARM
    value: "${PREWARM}"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
  {{- if .Values.config.file }}
    value: "${BASE_PATH}/ml_infra_core/llm_engine.core/llm_engine.core/configs/{{ .Values.config.file.infra }}"
  {{- else }}
    value: "${BASE_PATH}/ml_infra_core/llm_engine.core/llm_engine.core/configs/config.yaml"
  {{- end }}
{{- end }}

{{- define "llmEngine.syncServiceTemplateEnv" -}}
{{- include "llmEngine.baseServiceTemplateEnv" . }}
  - name: PORT
    value: "${ARTIFACT_LIKE_CONTAINER_PORT}"
{{- end }}

{{- define "llmEngine.asyncServiceTemplateEnv" -}}
{{- include "llmEngine.baseServiceTemplateEnv" . }}
  - name: CELERY_S3_BUCKET
    value: "${CELERY_S3_BUCKET}"
  - name: BROKER_TYPE
    value: "${BROKER_TYPE}"
  - name: SQS_PROFILE
    value: "${SQS_PROFILE}"
  - name: SQS_QUEUE_NAME
    value: "${QUEUE}"
  - name: SQS_QUEUE_URL
    value: "${SQS_QUEUE_URL}"
{{- end }}

{{- define "llmEngine.baseForwarderTemplateEnv" -}}
env:
  - name: DATADOG_TRACE_ENABLED
    value: "${DATADOG_TRACE_ENABLED}"
  - name: DD_SERVICE
    value: "${ENDPOINT_NAME}"
  - name: DD_ENV
    value: {{ .Values.context }}
  - name: DD_VERSION
    value: {{ .Values.tag }}
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: AWS_PROFILE
    value: "${AWS_ROLE}"
  - name: RESULTS_S3_BUCKET
    value: "${RESULTS_S3_BUCKET}"
  - name: BASE_PATH
    value: "/workspace"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
  {{- if .Values.config.file }}
    value: "/workspace/ml_infra_core/llm_engine.core/llm_engine.core/configs/{{ .Values.config.file.infra }}"
  {{- else }}
    value: "/workspace/ml_infra_core/llm_engine.core/llm_engine.core/configs/config.yaml"
  {{- end }}
{{- end }}

{{- define "llmEngine.syncForwarderTemplateEnv" -}}
{{- include "llmEngine.baseForwarderTemplateEnv" . }}
{{- if and .Values.forwarder .Values.forwarder.forceUseIPv4 }}
  - name: HTTP_HOST
    value: "0.0.0.0"
{{- end }}
{{- end }}

{{- define "llmEngine.asyncForwarderTemplateEnv" -}}
{{- include "llmEngine.baseForwarderTemplateEnv" . }}
  - name: CELERY_QUEUE
    value: "${QUEUE}"
  - name: CELERY_TASK_VISIBILITY
    value: "VISIBILITY_24H"
  - name: S3_BUCKET
    value: "${CELERY_S3_BUCKET}"
{{- end }}

{{- define "llmEngine.serviceEnv" }}
env:
  - name: DATADOG_TRACE_ENABLED
    value: "{{ .Values.datadog_trace_enabled }}"
  - name: DD_ENV
    value: {{ .Values.context }}
  - name: DD_VERSION
    value: {{ .Values.tag }}
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: GIT_TAG
    value: {{ .Values.tag }}
  - name: SERVICE_IDENTIFIER
    {{- if .Values.serviceIdentifier }}
    value: {{ .Values.serviceIdentifier }}
    {{- end }}
  {{- if .Values.aws }}
  - name: AWS_PROFILE
    value: {{ .Values.aws.profileName }}
  - name: ECR_READ_AWS_PROFILE
    value: {{ .Values.aws.profileName }}
  {{- end }}
  {{- with .Values.secrets }}
  {{- if .kubernetesDatabaseSecretName }}
  - name: ML_INFRA_DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: {{ .kubernetesDatabaseSecretName }}
        key: database_url
  {{- else if .awsDatabaseSecretName }}
  - name: DB_SECRET_NAME
    value: {{ .awsDatabaseSecretName }}
  {{- end }}
  {{- end }}
  {{- if .Values.config.file }}
  - name: DEPLOY_SERVICE_CONFIG_PATH
    value: "/workspace/llm_engine/service_configs/{{ .Values.config.file.llm_engine }}"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
    value: "/workspace/ml_infra_core/llm_engine.core/llm_engine.core/configs/{{ .Values.config.file.infra }}"
  {{- else }}
  - name: DEPLOY_SERVICE_CONFIG_PATH
    value: "/workspace/llm_engine/service_configs/service_config.yaml"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
    value: "/workspace/ml_infra_core/llm_engine.core/llm_engine.core/configs/config.yaml"
  {{- end }}
  - name: CELERY_ELASTICACHE_ENABLED
    value: "true"
  - name: LLM_ENGINE_SERVICE_TEMPLATE_FOLDER
    value: "/workspace/llm_engine/llm_engine/infra/gateways/resources/templates"
{{- end }}

{{- define "llmEngine.gatewayEnv" }}
{{- include "llmEngine.serviceEnv" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "llmEngine.fullname" .) }}
{{- end }}

{{- define "llmEngine.builderEnv" }}
{{- include "llmEngine.serviceEnv" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "llmEngine.buildername" .) }}
{{- end }}

{{- define "llmEngine.cacherEnv" }}
{{- include "llmEngine.serviceEnv" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "llmEngine.cachername" .) }}
{{- end }}

{{- define "llmEngine.volumes" }}
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
  - name: service-template-config
    configMap:
      name: {{ include "llmEngine.fullname" . }}-service-template-config
  {{- if .Values.aws }}
  - name: config-volume
    configMap:
      name: {{ .Values.aws.configMap.name }}
  {{- end }}
  {{- if .Values.config.values }}
  - name: llm-engine-service-config-volume
    configMap:
      name: {{ include "llmEngine.fullname" . }}-service-config
      items:
        - key: llm_engine_service_config
          path: service_config.yaml
  - name: infra-service-config-volume
    configMap:
      name: {{ include "llmEngine.fullname" . }}-service-config
      items:
        - key: infra_service_config
          path: config.yaml
  {{- end }}
{{- end }}

{{- define "llmEngine.volumeMounts" }}
volumeMounts:
  - name: dshm
    mountPath: /dev/shm
  - name: service-template-config
    mountPath: /workspace/llm_engine/llm_engine/infra/gateways/resources/templates
  {{- if .Values.aws }}
  - name: config-volume
    mountPath: /home/user/.aws/config
    subPath: config
  {{- end }}
  {{- if .Values.config.values }}
  - name: llm-engine-service-config-volume
    mountPath: /workspace/llm_engine/service_configs
  - name: infra-service-config-volume
    mountPath: /workspace/ml_infra_core/llm_engine.core/llm_engine.core/configs
  {{- end }}
{{- end }}

{{- define "llmEngine.forwarderVolumeMounts" }}
volumeMounts:
  - name: config-volume
    mountPath: /root/.aws/config
    subPath: config
  - name: user-config
    mountPath: /workspace/user_config
    subPath: raw_data
  - name: endpoint-config
    mountPath: /workspace/endpoint_config
    subPath: raw_data
  {{- if .Values.config.values }}
  - name: infra-service-config-volume
    mountPath: /workspace/ml_infra_core/llm_engine.core/llm_engine.core/configs
  {{- end }}
{{- end }}

{{- define "llmEngine.serviceAccountNamespaces" }}
namespaces:
  - {{ .Release.Namespace }}
{{- range .Values.serviceAccount.namespaces }}
  - {{ . }}
{{- end }}
{{- end }}
