{{/*
Expand the name of the chart.
*/}}
{{- define "launch.name" -}}
{{- default .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 40 chars because some Kubernetes name fields are limited to 63 (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "launch.fullname" -}}
{{- if .Values.serviceIdentifier }}
{{- printf "%s-%s" .Chart.Name .Values.serviceIdentifier | trunc 40 | trimSuffix "-" }}
{{- else }}
{{- default .Chart.Name | trunc 40 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "launch.buildername" -}}
"{{ include "launch.fullname" . }}-endpoint-builder"
{{- end }}

{{- define "launch.cachername" -}}
"{{ include "launch.fullname" . }}-cacher"
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "launch.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "launch.labels" -}}
team: infra
product: launch
helm.sh/chart: {{ include "launch.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Values.tag }}
tags.datadoghq.com/version: {{ .Values.tag }}
tags.datadoghq.com/env: {{ .Values.context }}
{{- end }}

{{- define "launch.selectorLabels.builder" -}}
app: {{ include "launch.buildername" . }}
{{- end }}

{{- define "launch.selectorLabels.cacher" -}}
app: {{ include "launch.cachername" . }}
{{- end }}

{{- define "launch.selectorLabels.gateway" -}}
app: {{ include "launch.fullname" . -}}
{{- end }}

{{- define "launch.baseTemplateLabels" -}}
user_id: ${OWNER}
team: ${TEAM}
product: ${PRODUCT}
created_by: ${CREATED_BY}
owner: ${OWNER}
env: {{- .Values.context | printf " %s" }}
managed-by: {{- include "launch.fullname" . | printf " %s\n" -}}
use_scale_launch_endpoint_network_policy: "true"
tags.datadoghq.com/env: {{- .Values.context | printf " %s" }}
tags.datadoghq.com/version: {{- .Values.tag | printf " %s" }}
{{- end }}

{{- define "launch.serviceTemplateLabels" -}}
{{- include "launch.baseTemplateLabels" . | printf "%s\n" -}}
tags.datadoghq.com/service: ${ENDPOINT_NAME}
endpoint_id: ${ENDPOINT_ID}
endpoint_name: ${ENDPOINT_NAME}
{{- end }}

{{- define "launch.jobTemplateLabels" -}}
{{- include "launch.baseTemplateLabels" . | printf "%s\n" -}}
launch_job_id: ${JOB_ID}
tags.datadoghq.com/service: ${JOB_ID}
{{- end }}

{{- define "launch.serviceTemplateAsyncAnnotations" -}}
celery.scaleml.autoscaler/queue: ${QUEUE}
celery.scaleml.autoscaler/broker: ${BROKER_NAME}
celery.scaleml.autoscaler/taskVisibility: "VISIBILITY_24H"
celery.scaleml.autoscaler/perWorker: "${PER_WORKER}"
celery.scaleml.autoscaler/minWorkers: "${MIN_WORKERS}"
celery.scaleml.autoscaler/maxWorkers: "${MAX_WORKERS}"
{{- end }}

{{- define "launch.serviceTemplateAffinity" -}}
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

{{- define "launch.baseServiceTemplateEnv" -}}
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
    value: "${BASE_PATH}/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/{{ .Values.config.file.infra }}"
  {{- else }}
    value: "${BASE_PATH}/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/config.yaml"
  {{- end }}
{{- end }}

{{- define "launch.syncServiceTemplateEnv" -}}
{{- include "launch.baseServiceTemplateEnv" . }}
  - name: PORT
    value: "${ARTIFACT_LIKE_CONTAINER_PORT}"
{{- end }}

{{- define "launch.asyncServiceTemplateEnv" -}}
{{- include "launch.baseServiceTemplateEnv" . }}
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

{{- define "launch.baseForwarderTemplateEnv" -}}
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
    value: "/workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/{{ .Values.config.file.infra }}"
  {{- else }}
    value: "/workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/config.yaml"
  {{- end }}
{{- end }}

{{- define "launch.syncForwarderTemplateEnv" -}}
{{- include "launch.baseForwarderTemplateEnv" . }}
{{- if and .Values.forwarder .Values.forwarder.forceUseIPv4 }}
  - name: HTTP_HOST
    value: "0.0.0.0"
{{- end }}
{{- end }}

{{- define "launch.asyncForwarderTemplateEnv" -}}
{{- include "launch.baseForwarderTemplateEnv" . }}
  - name: CELERY_QUEUE
    value: "${QUEUE}"
  - name: CELERY_TASK_VISIBILITY
    value: "VISIBILITY_24H"
  - name: S3_BUCKET
    value: "${CELERY_S3_BUCKET}"
{{- end }}

{{- define "launch.serviceEnv" }}
env:
  - name: DATADOG_TRACE_ENABLED
    value: "true"
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
    value: "/workspace/spellbook_serve/service_configs/{{ .Values.config.file.launch }}"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
    value: "/workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/{{ .Values.config.file.infra }}"
  {{- else }}
  - name: DEPLOY_SERVICE_CONFIG_PATH
    value: "/workspace/spellbook_serve/service_configs/service_config.yaml"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
    value: "/workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/config.yaml"
  {{- end }}
  - name: CELERY_ELASTICACHE_ENABLED
    value: "true"
  - name: LAUNCH_SERVICE_TEMPLATE_FOLDER
    value: "/workspace/spellbook_serve/spellbook_serve/infra/gateways/resources/templates"
{{- end }}

{{- define "launch.gatewayEnv" }}
{{- include "launch.serviceEnv" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "launch.fullname" .) }}
{{- end }}

{{- define "launch.builderEnv" }}
{{- include "launch.serviceEnv" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "launch.buildername" .) }}
{{- end }}

{{- define "launch.cacherEnv" }}
{{- include "launch.serviceEnv" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "launch.cachername" .) }}
{{- end }}

{{- define "launch.volumes" }}
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
  - name: service-template-config
    configMap:
      name: {{ include "launch.fullname" . }}-service-template-config
  {{- if .Values.aws }}
  - name: config-volume
    configMap:
      name: {{ .Values.aws.configMap.name }}
  {{- end }}
  {{- if .Values.config.values }}
  - name: launch-service-config-volume
    configMap:
      name: {{ include "launch.fullname" . }}-service-config
      items:
        - key: launch_service_config
          path: service_config.yaml
  - name: infra-service-config-volume
    configMap:
      name: {{ include "launch.fullname" . }}-service-config
      items:
        - key: infra_service_config
          path: config.yaml
  {{- end }}
{{- end }}

{{- define "launch.volumeMounts" }}
volumeMounts:
  - name: dshm
    mountPath: /dev/shm
  - name: service-template-config
    mountPath: /workspace/spellbook_serve/spellbook_serve/infra/gateways/resources/templates
  {{- if .Values.aws }}
  - name: config-volume
    mountPath: /root/.aws/config
    subPath: config
  {{- end }}
  {{- if .Values.config.values }}
  - name: launch-service-config-volume
    mountPath: /workspace/spellbook_serve/service_configs
  - name: infra-service-config-volume
    mountPath: /workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs
  {{- end }}
{{- end }}

{{- define "launch.forwarderVolumeMounts" }}
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
    mountPath: /workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs
  {{- end }}
{{- end }}

{{- define "launch.serviceAccountNamespaces" }}
namespaces:
  - {{ .Release.Namespace }}
{{- range .Values.serviceAccount.namespaces }}
  - {{ . }}
{{- end }}
{{- end }}
