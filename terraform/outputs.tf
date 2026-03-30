output "dvc_bucket_name" {
  description = "The name of the dynamically generated DVC remote storage S3 bucket."
  value       = aws_s3_bucket.dvc_storage.bucket
}

output "us_east_service_url" {
  description = "US East App Runner Service URL"
  value       = aws_apprunner_service.mlops_api_us_east.service_url
}

output "eu_west_service_url" {
  description = "EU West App Runner Service URL"
  value       = aws_apprunner_service.mlops_api_eu_west.service_url
}
