{{/*
Expand the name of the chart.
*/}}
{{- define "modelEngine.name" -}}
{{- default .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 40 chars because some Kubernetes name fields are limited to 63 (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "modelEngine.fullname" -}}
{{- if .Values.serviceIdentifier }}
{{- printf "%s-%s" .Chart.Name .Values.serviceIdentifier | trunc 40 | trimSuffix "-" }}
{{- else }}
{{- default .Chart.Name | trunc 40 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "modelEngine.buildername" -}}
"{{ include "modelEngine.fullname" . }}-endpoint-builder"
{{- end }}

{{- define "modelEngine.cachername" -}}
"{{ include "modelEngine.fullname" . }}-cacher"
{{- end }}

{{- define "modelEngine.gatewayurl" -}}
{{ .Values.hostDomain.prefix }}{{ include "modelEngine.fullname" . }}.{{ .Release.Namespace }}:{{ .Values.service.port }}
{{- end }}

{{- define "modelEngine.celeryautoscalername" -}}
{{- if .Values.serviceIdentifier }}
{{- printf "celery-autoscaler-%s-%s" .Values.celeryBrokerType .Values.serviceIdentifier }}
{{- else }}
{{- printf "celery-autoscaler-%s" .Values.celeryBrokerType }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "modelEngine.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "modelEngine.baseLabels" -}}
team: infra
app.kubernetes.io/version: {{ .Values.tag }}
tags.datadoghq.com/version: {{ .Values.tag }}
tags.datadoghq.com/env: {{ .Values.context }}
env: {{ .Values.context }}
{{- if .Values.azure }}
azure.workload.identity/use: "true"
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "modelEngine.labels" -}}
{{- include "modelEngine.baseLabels" . | printf "%s\n" -}}
product: model-engine
helm.sh/chart: {{ include "modelEngine.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "modelEngine.selectorLabels.builder" -}}
app: {{ include "modelEngine.buildername" . }}
{{- end }}

{{- define "modelEngine.selectorLabels.cacher" -}}
app: {{ include "modelEngine.cachername" . }}
{{- end }}

{{- define "modelEngine.selectorLabels.gateway" -}}
app: {{ include "modelEngine.fullname" . -}}
{{- end }}

{{- define "modelEngine.selectorLabels.celeryAutoscaler" -}}
app: {{ include "modelEngine.celeryautoscalername" . }}
product: common
tags.datadoghq.com/service: {{ include "modelEngine.celeryautoscalername" . -}}
{{- end }}

{{- define "modelEngine.baseTemplateLabels" -}}
user_id: ${OWNER}
team: ${TEAM}
product: ${PRODUCT}
created_by: ${CREATED_BY}
owner: ${OWNER}
env: {{- .Values.context | printf " %s" }}
managed-by: {{- include "modelEngine.fullname" . | printf " %s\n" -}}
use_scale_launch_endpoint_network_policy: "true"
tags.datadoghq.com/env: {{- .Values.context | printf " %s" }}
tags.datadoghq.com/version: ${GIT_TAG}
{{- if .Values.azure }}
azure.workload.identity/use: "true"
{{- end }}
{{- end }}

{{- define "modelEngine.serviceTemplateLabels" -}}
{{- include "modelEngine.baseTemplateLabels" . | printf "%s\n" -}}
tags.datadoghq.com/service: ${ENDPOINT_NAME}
endpoint_id: ${ENDPOINT_ID}
endpoint_name: ${ENDPOINT_NAME}
{{- end }}

{{- define "modelEngine.jobTemplateLabels" -}}
{{- include "modelEngine.baseTemplateLabels" . | printf "%s\n" -}}
launch_job_id: ${JOB_ID}
tags.datadoghq.com/request_id: ${REQUEST_ID}
tags.datadoghq.com/service: ${JOB_ID}
tags.datadoghq.com/user_id: ${OWNER}
tags.datadoghq.com/team: ${TEAM}
{{- end }}

{{- define "modelEngine.serviceTemplateAsyncAnnotations" -}}
celery.scaleml.autoscaler/queue: ${QUEUE}
celery.scaleml.autoscaler/broker: ${BROKER_NAME}
celery.scaleml.autoscaler/taskVisibility: "VISIBILITY_24H"
celery.scaleml.autoscaler/perWorker: "${PER_WORKER}"
celery.scaleml.autoscaler/minWorkers: "${MIN_WORKERS}"
celery.scaleml.autoscaler/maxWorkers: "${MAX_WORKERS}"
{{- end }}

{{- define "modelEngine.serviceTemplateAffinity" -}}
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

{{- define "modelEngine.baseServiceTemplateEnv" -}}
env:
  - name: DD_TRACE_ENABLED
    value: "${DD_TRACE_ENABLED}"
  - name: DD_REMOTE_CONFIGURATION_ENABLED
    value: "false"
  - name: DD_SERVICE
    value: "${ENDPOINT_NAME}"
  - name: DD_ENV
    value: {{ .Values.context }}
  - name: DD_VERSION
    value: "${GIT_TAG}"
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
  {{- if .Values.aws }}
  - name: AWS_PROFILE
    value: "${AWS_ROLE}"
  {{- end }}
  - name: RESULTS_S3_BUCKET
    value: "${RESULTS_S3_BUCKET}"
  - name: CHILD_FN_INFO
    value: "${CHILD_FN_INFO}"
  - name: PREWARM
    value: "${PREWARM}"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
  {{- if .Values.config.file }}
    value: {{ .Values.config.file.infra | quote }}
  {{- else }}
    value: "${BASE_PATH}/model-engine/model_engine_server/core/configs/config.yaml"
  {{- end }}
{{- end }}

{{- define "modelEngine.syncServiceTemplateEnv" -}}
{{- include "modelEngine.baseServiceTemplateEnv" . }}
  - name: PORT
    value: "${ARTIFACT_LIKE_CONTAINER_PORT}"
{{- end }}

{{- define "modelEngine.asyncServiceTemplateEnv" -}}
{{- include "modelEngine.baseServiceTemplateEnv" . }}
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

{{- define "modelEngine.baseForwarderTemplateEnv" -}}
env:
  - name: DD_TRACE_ENABLED
    value: "${DD_TRACE_ENABLED}"
  - name: DD_REMOTE_CONFIGURATION_ENABLED
    value: "false"
  - name: DD_SERVICE
    value: "${ENDPOINT_NAME}"
  - name: DD_ENV
    value: {{ .Values.context }}
  - name: DD_VERSION
    value: "${GIT_TAG}"
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  {{- if .Values.aws }}
  - name: AWS_PROFILE
    value: "${AWS_ROLE}"
  - name: AWS_CONFIG_FILE
    value: /opt/.aws/config
  {{- end }}
  - name: RESULTS_S3_BUCKET
    value: "${RESULTS_S3_BUCKET}"
  - name: BASE_PATH
    value: "/workspace"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
  {{- if .Values.config.file }}
    value: {{ .Values.config.file.infra | quote }}
  {{- else }}
    value: "/workspace/model-engine/model_engine_server/core/configs/config.yaml"
  {{- end }}
  {{- if .Values.azure}}
  - name: AZURE_IDENTITY_NAME
    value: {{ .Values.azure.identity_name }}
  - name: AZURE_CLIENT_ID
    value: {{ .Values.azure.client_id }}
  - name: AZURE_OBJECT_ID
    value: {{ .Values.azure.object_id }}
  - name: ABS_ACCOUNT_NAME
    value: {{ .Values.azure.abs_account_name }}
  - name: ABS_CONTAINER_NAME
    value: {{ .Values.azure.abs_container_name }}
  {{- end }}
{{- end }}

{{- define "modelEngine.syncForwarderTemplateEnv" -}}
{{- include "modelEngine.baseForwarderTemplateEnv" . }}
{{- if and .Values.forwarder .Values.forwarder.forceUseIPv4 }}
  - name: HTTP_HOST
    value: "0.0.0.0"
{{- end }}
{{- end }}

{{- define "modelEngine.asyncForwarderTemplateEnv" -}}
{{- include "modelEngine.baseForwarderTemplateEnv" . }}
  - name: CELERY_QUEUE
    value: "${QUEUE}"
  - name: CELERY_TASK_VISIBILITY
    value: "VISIBILITY_24H"
  - name: S3_BUCKET
    value: "${CELERY_S3_BUCKET}"
  {{- if .Values.azure}}
  - name: ABS_ACCOUNT_NAME
    value: {{ .Values.azure.abs_account_name }}
  - name: ABS_CONTAINER_NAME
    value: {{ .Values.azure.abs_container_name }}
  - name: SERVICEBUS_NAMESPACE
    value: {{ .Values.azure.servicebus_namespace }}
  {{- end }}
{{- end }}

{{- define "modelEngine.serviceEnvBase" }}
env:
  - name: DD_TRACE_ENABLED
    value: "{{ .Values.dd_trace_enabled }}"
  - name: DD_REMOTE_CONFIGURATION_ENABLED
    value: "false"
  - name: DD_ENV
    value: {{ .Values.context }}
  - name: DD_AGENT_HOST
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: SERVICE_IDENTIFIER
    {{- if .Values.serviceIdentifier }}
    value: {{ .Values.serviceIdentifier }}
    {{- end }}
  - name: SERVICE_BUILDER_QUEUE
    {{- if .Values.serviceBuilderQueue }}
    value: {{ .Values.serviceBuilderQueue }}
    {{- end }}
  - name: GATEWAY_URL
    value: {{ include "modelEngine.gatewayurl" . }}
  {{- if .Values.aws }}
  - name: AWS_PROFILE
    value: {{ .Values.aws.profileName }}
  - name: AWS_CONFIG_FILE
    value: /opt/.aws/config
  - name: ECR_READ_AWS_PROFILE
    value: {{ .Values.aws.profileName }}
  - name: DB_SECRET_AWS_PROFILE
    value: {{ .Values.aws.profileName }}
  - name: S3_WRITE_AWS_PROFILE
    value: {{ .Values.aws.s3WriteProfileName }}
  {{- end }}
  {{- with .Values.secrets }}
  {{- if .kubernetesDatabaseSecretName }}
  - name: ML_INFRA_DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: {{ .kubernetesDatabaseSecretName }}
        key: database_url
  {{- else if .cloudDatabaseSecretName }}
  - name: DB_SECRET_NAME
    value: {{ .cloudDatabaseSecretName }}
  {{- end }}
  {{- end }}
  {{- if .Values.config.file }}
  - name: DEPLOY_SERVICE_CONFIG_PATH
    value: {{ .Values.config.file.launch | quote }}
  - name: ML_INFRA_SERVICES_CONFIG_PATH
    value: {{ .Values.config.file.infra | quote }}
  {{- else }}
  - name: DEPLOY_SERVICE_CONFIG_PATH
    value: "/workspace/model-engine/service_configs/service_config.yaml"
  - name: ML_INFRA_SERVICES_CONFIG_PATH
    value: "/workspace/model-engine/model_engine_server/core/configs/config.yaml"
  {{- end }}
  - name: CELERY_ELASTICACHE_ENABLED
    value: "true"
  - name: LAUNCH_SERVICE_TEMPLATE_FOLDER
    value: "/workspace/model-engine/model_engine_server/infra/gateways/resources/templates"
  {{- if .Values.redis.auth}}
  - name: REDIS_AUTH_TOKEN
    value: {{ .Values.redis.auth }}
  {{- end }}
  {{- if .Values.azure}}
  - name: AZURE_IDENTITY_NAME
    value: {{ .Values.azure.identity_name }}
  - name: AZURE_CLIENT_ID
    value: {{ .Values.azure.client_id }}
  - name: AZURE_OBJECT_ID
    value: {{ .Values.azure.object_id }}
  - name: KEYVAULT_NAME
    value: {{ .Values.azure.keyvault_name }}
  - name: ABS_ACCOUNT_NAME
    value: {{ .Values.azure.abs_account_name }}
  - name: ABS_CONTAINER_NAME
    value: {{ .Values.azure.abs_container_name }}
  - name: SERVICEBUS_NAMESPACE
    value: {{ .Values.azure.servicebus_namespace }}
  {{- end }}
  {{- if eq .Values.context "circleci" }}
  - name: CIRCLECI
    value: "true"
  {{- end }}
  {{- if .Values.gunicorn }}
  - name: WORKER_TIMEOUT
    value: {{ .Values.gunicorn.workerTimeout | quote }}
  - name: GUNICORN_TIMEOUT
    value: {{ .Values.gunicorn.gracefulTimeout | quote }}
  - name: GUNICORN_GRACEFUL_TIMEOUT
    value: {{ .Values.gunicorn.gracefulTimeout | quote }}
  - name: GUNICORN_KEEP_ALIVE
    value: {{ .Values.gunicorn.keepAlive | quote }}
  - name: GUNICORN_WORKER_CLASS
    value: {{ .Values.gunicorn.workerClass | quote }}
  {{- end }}
{{- end }}

{{- define "modelEngine.serviceEnvGitTagFromHelmVar" }}
{{- include "modelEngine.serviceEnvBase" . }}
  - name: DD_VERSION
    value: {{ .Values.tag }}
  - name: GIT_TAG
    value: {{ .Values.tag }}
{{- end }}

{{- define "modelEngine.serviceEnvGitTagFromPythonReplace" }}
{{- include "modelEngine.serviceEnvBase" . }}
  - name: DD_VERSION
    value: "${GIT_TAG}"
  - name: GIT_TAG
    value: "${GIT_TAG}"
{{- end }}


{{- define "modelEngine.gatewayEnv" }}
{{- include "modelEngine.serviceEnvGitTagFromHelmVar" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "modelEngine.fullname" .) }}
{{- end }}

{{- define "modelEngine.builderEnv" }}
{{- include "modelEngine.serviceEnvGitTagFromHelmVar" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "modelEngine.buildername" .) }}
{{- end }}

{{- define "modelEngine.cacherEnv" }}
{{- include "modelEngine.serviceEnvGitTagFromHelmVar" . }}
  - name: DD_SERVICE
    value: {{- printf " %s" (include "modelEngine.cachername" .) }}
{{- end }}

{{- define "modelEngine.volumes" }}
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
  - name: service-template-config
    configMap:
      name: {{ include "modelEngine.fullname" . }}-service-template-config
  {{- if .Values.aws }}
  - name: config-volume
    configMap:
      name: {{ .Values.aws.configMap.name }}
  {{- end }}
  {{- if .Values.config.values }}
  - name: {{ .Chart.Name }}-service-config-volume
    configMap:
      name: {{ include "modelEngine.fullname" . }}-service-config
      items:
        - key: launch_service_config
          path: service_config.yaml
  - name: infra-service-config-volume
    configMap:
      name: {{ include "modelEngine.fullname" . }}-service-config
      items:
        - key: infra_service_config
          path: config.yaml
  {{- end }}
{{- end }}

{{- define "modelEngine.volumeMounts" }}
volumeMounts:
  - name: dshm
    mountPath: /dev/shm
  - name: service-template-config
    mountPath: /workspace/model-engine/model_engine_server/infra/gateways/resources/templates
  {{- if .Values.aws }}
  - name: config-volume
    mountPath: {{ .Values.aws.configMap.mountPath }}
    subPath: config
  {{- end }}
  {{- if .Values.config.values }}
  - name: {{ .Chart.Name }}-service-config-volume
    mountPath: /workspace/model-engine/service_configs
  - name: infra-service-config-volume
    mountPath: /workspace/model-engine/model_engine_server/core/configs
  {{- end }}
{{- end }}

{{- define "modelEngine.forwarderVolumeMounts" }}
volumeMounts:
  {{- if .Values.aws }}
  - name: config-volume
    mountPath: /opt/.aws/config
    subPath: config
  {{- end }}
  - name: user-config
    mountPath: /workspace/user_config
    subPath: raw_data
  - name: endpoint-config
    mountPath: /workspace/endpoint_config
    subPath: raw_data
  {{- if .Values.config.values }}
  - name: infra-service-config-volume
    mountPath: /workspace/model-engine/model_engine_server/core/configs
  {{- end }}
{{- end }}

{{- define "modelEngine.serviceAccountNamespaces" }}
namespaces:
  - {{ .Release.Namespace }}
{{- range .Values.serviceAccount.namespaces }}
  - {{ . }}
{{- end }}
{{- end }}
