terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_regions" {
  description = "List of AWS regions to deploy the App Runner service to"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1"]
}

variable "project_name" {
  description = "The root name of the project"
  type        = string
  default     = "cars-mlops-prediction"
}

variable "ecr_image_uri" {
  description = "The URI of the Docker image in Amazon ECR"
  type        = string
}

variable "app_runner_instance_role_arn" {
  description = "The IAM Role ARN for App Runner to access the ECR registry"
  type        = string
}
