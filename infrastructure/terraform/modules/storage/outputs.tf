output "artifacts_bucket_name" {
  value = google_storage_bucket.artifacts.name
}

output "data_bucket_name" {
  value = google_storage_bucket.data.name
}

output "mlflow_bucket_name" {
  value = google_storage_bucket.mlflow.name
}
