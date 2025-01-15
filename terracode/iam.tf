# Get the current project
data "google_project" "current" {}

### FROM CLAUDE

# Create the Workload Identity Pool
resource "google_iam_workload_identity_pool" "gke_pool" {
  project                   = var.project_id
  workload_identity_pool_id = "gke-pool"
  display_name             = "GKE Workload Identity Pool"
  description              = "Identity pool for GKE workload identity"
}

# Create Workload Identity Pool Provider for GKE
resource "google_iam_workload_identity_pool_provider" "gke_provider" {
  project                            = var.project_id
  workload_identity_pool_id         = google_iam_workload_identity_pool.gke_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "gke-provider"
  display_name                      = "GKE Provider"
  description                       = "Workload Identity Pool Provider for GKE"
  
  attribute_mapping = {
    "google.subject"                        = "assertion.sub"
    "attribute.aud"                         = "assertion.aud"
    "attribute.namespace"    = "assertion.namespace"
    "attribute.service_account" = "assertion.kubernetes.serviceaccount.name"
  }

  oidc {
    allowed_audiences = ["https://kubernetes.default.svc.cluster.local"]
    issuer_uri       = "https://container.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/clusters/${var.cluster_name}"
  }
}

# Update your existing service account IAM bindings
# Update llm_engine service account IAM binding
resource "google_service_account_iam_binding" "llm_engine_workload_identity" {
  service_account_id = google_service_account.llm_engine.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.gke_pool.name}/attribute.namespace/default/attribute.service_account/llm-engine"
  ]

  depends_on = [
    google_iam_workload_identity_pool.gke_pool,
    google_iam_workload_identity_pool_provider.gke_provider
  ]
}

# Update kaniko service account IAM binding
resource "google_service_account_iam_binding" "kaniko_workload_identity" {
  service_account_id = google_service_account.kaniko.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.gke_pool.name}/attribute.namespace/default/attribute.service_account/kaniko"
  ]

  depends_on = [
    google_iam_workload_identity_pool.gke_pool,
    google_iam_workload_identity_pool_provider.gke_provider
  ]
}


####################
# LLM Engine Resources
####################

# Create service account for LLM engine
resource "google_service_account" "llm_engine" {
  account_id   = "${local.prefix}-llm-engine"
  display_name = "LLM Engine Service Account"
}

# Grant Storage permissions for ML bucket
resource "google_storage_bucket_iam_member" "llm_engine_ml_bucket" {
  bucket = google_storage_bucket.ml_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.llm_engine.email}"
}

resource "google_storage_bucket_iam_member" "llm_engine_ml_bucket_write" {
  bucket = google_storage_bucket.ml_bucket.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.llm_engine.email}"
}

# Grant Storage permissions for models bucket
resource "google_storage_bucket_iam_member" "llm_engine_models_bucket" {
  bucket = google_storage_bucket.models_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.llm_engine.email}"
}

# Grant Pub/Sub permissions (equivalent to SQS)
resource "google_project_iam_member" "llm_engine_pubsub" {
  project = data.google_project.current.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.llm_engine.email}"
}

resource "google_project_iam_member" "llm_engine_pubsub_subscriber" {
  project = data.google_project.current.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.llm_engine.email}"
}

# Grant Artifact Registry permissions (equivalent to ECR)
resource "google_project_iam_member" "llm_engine_artifact_registry" {
  project = data.google_project.current.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.llm_engine.email}"
}

####################
# Image Builder Resources
####################

# Create service account for Kaniko
resource "google_service_account" "kaniko" {
  account_id   = "${local.prefix}-kaniko"
  display_name = "Kaniko Service Account"
}

# Grant Artifact Registry permissions for Kaniko
resource "google_project_iam_member" "kaniko_artifact_registry" {
  project = data.google_project.current.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.kaniko.email}"
}

# Grant Storage permissions for Kaniko
resource "google_storage_bucket_iam_member" "kaniko_storage" {
  bucket = google_storage_bucket.ml_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.kaniko.email}"
}

# KMS equivalent (if needed)
resource "google_project_iam_member" "llm_engine_kms" {
  count   = var.use_cmk ? 1 : 0
  project = data.google_project.current.project_id
  role    = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member  = "serviceAccount:${google_service_account.llm_engine.email}"
}

resource "google_project_iam_member" "kaniko_kms" {
  count   = var.use_cmk ? 1 : 0
  project = data.google_project.current.project_id
  role    = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member  = "serviceAccount:${google_service_account.kaniko.email}"
}

# Secret Manager access (equivalent to Redis secrets)
resource "google_secret_manager_secret_iam_member" "llm_engine_redis_secret" {
  count     = var.use_redis_secret ? 1 : 0
  secret_id = var.redis_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.llm_engine.email}"
}