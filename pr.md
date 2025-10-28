# Add On-Premise Deployment Support

This PR adds comprehensive support for on-premise deployments using Redis, MinIO/S3-compatible storage, and private container registries as alternatives to cloud-managed services.

## Key Changes

- **New on-prem configuration**: Added `onprem.yaml` config file with settings for MinIO, Redis, and private registries
- **Redis-based infrastructure**: Implemented Redis task queues and on-prem queue endpoint delegate
- **S3-compatible storage**: Added support for MinIO and custom S3 endpoints with configurable addressing styles
- **Container registry flexibility**: Support for private registries with `OnPremDockerRepository`
- **Database configuration**: Environment variable-based PostgreSQL connection for on-prem deployments
- **Improved logging**: Enhanced error handling and debug logs in S3 file storage gateway

## Configuration Highlights

The on-prem setup allows deployments to use:
- MinIO or S3-compatible object storage instead of AWS S3/Azure Blob
- Redis for Celery task queues and caching instead of SQS/ASB
- Local PostgreSQL with environment-based credentials
- Private container registries instead of ECR/ACR
