{{- define "serve.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "serve.fullname" -}}
{{- .Release.Name -}}-{{- .Chart.Name -}}
{{- end -}}

{{- define "serve.labels" -}}
app.kubernetes.io/name: {{ include "serve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
