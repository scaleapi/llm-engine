variable "use_cmk" {
  type    = bool
  default = false
}

variable "use_redis_secret" {
  type    = bool
  default = false
}

variable "redis_secret_id" {
  type    = string
  default = ""
}

# Add bucket resources
resource "google_storage_bucket" "ml_bucket" {
  name     = "${local.prefix}-ml-bucket"
  location = "us-central1"
}

resource "google_storage_bucket" "models_bucket" {
  name     = "${local.prefix}-models-bucket"
  location = "us-central1"
}

# Variables you'll need to define
variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "sgp-model-engine-on-gcp-test"
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "The name of your GKE cluster"
  type        = string
  default     = "my-gke-cluster"
}