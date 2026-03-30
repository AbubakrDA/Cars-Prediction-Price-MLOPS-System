provider "aws" {
  region = "us-east-1"
  alias  = "us_east"
}

provider "aws" {
  region = "eu-west-1"
  alias  = "eu_west"
}

# --- Shared MLOps Infrastructure ---

# DVC Storage Bucket (Dynamically Generated)
resource "aws_s3_bucket" "dvc_storage" {
  provider = aws.us_east
  bucket   = "${var.project_name}-dvc-storage-${random_id.bucket_suffix.hex}"
  
  tags = {
    Environment = "Production"
    Purpose     = "DVC Remote Artifact Store"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# --- Multi-Region AWS App Runner Services ---

# US East 1 Instance
resource "aws_apprunner_service" "mlops_api_us_east" {
  provider     = aws.us_east
  service_name = "${var.project_name}-api-us-east"

  source_configuration {
    auto_deployments_enabled = false
    
    authentication_configuration {
      access_role_arn = var.app_runner_instance_role_arn
    }

    image_repository {
      image_identifier      = var.ecr_image_uri
      image_repository_type = "ECR"
      image_configuration {
        port = "8000"
      }
    }
  }

  instance_configuration {
    cpu    = "1024" # 1 vCPU
    memory = "2048" # 2 GB
  }
}

# EU West 1 Instance
resource "aws_apprunner_service" "mlops_api_eu_west" {
  provider     = aws.eu_west
  service_name = "${var.project_name}-api-eu-west"

  source_configuration {
    auto_deployments_enabled = false
    
    authentication_configuration {
      access_role_arn = var.app_runner_instance_role_arn
    }

    image_repository {
      image_identifier      = var.ecr_image_uri
      image_repository_type = "ECR"
      image_configuration {
        port = "8000"
      }
    }
  }

  instance_configuration {
    cpu    = "1024" # 1 vCPU
    memory = "2048" # 2 GB
  }
}
