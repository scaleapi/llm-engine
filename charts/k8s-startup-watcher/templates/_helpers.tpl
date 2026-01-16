{{/*
Expand the name of the chart.
*/}}
{{- define "startupWatcher.name" -}}
{{- default .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "startupWatcher.fullname" -}}
{{- default .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "startupWatcher.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "startupWatcher.labels" -}}
helm.sh/chart: {{ include "startupWatcher.chart" . }}
app.kubernetes.io/name: {{ include "startupWatcher.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
team: infra
product: ml-infra
env: {{ .Values.context }}
tags.datadoghq.com/env: {{ .Values.context }}
tags.datadoghq.com/service: {{ include "startupWatcher.fullname" . }}
tags.datadoghq.com/version: {{ .Values.image.tag | quote }}
{{- with .Values.extraLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "startupWatcher.selectorLabels" -}}
app.kubernetes.io/name: {{ include "startupWatcher.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: {{ include "startupWatcher.fullname" . }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "startupWatcher.serviceAccountName" -}}
{{ include "startupWatcher.fullname" . }}
{{- end }}
