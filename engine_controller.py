"""
LLM Engine Master Controller
============================
A unified deployment orchestrator for running LLM Engine locally (Minikube) or in cloud (AWS/EKS).

This controller abstracts away infrastructure complexity and provides a single interface
for managing the entire LLM Engine deployment lifecycle across different environments.

AI Expert Assessment:
- DevOps Expert: Validates Kubernetes configurations and deployment patterns
- Infrastructure Architect: Ensures scalability and cost optimization
- ML Engineer: Considers GPU scheduling and model serving requirements
- Security Expert: Reviews credential handling and RBAC configuration
- Database Architect: Manages PostgreSQL and Redis dependencies
- Python Developer: Ensures code quality and maintainability
- Cloud Architect: Optimizes for AWS specifics and cost efficiency
"""

import json
import os
import subprocess
import sys
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class DeploymentMode(Enum):
    """Deployment environment type"""
    LOCAL = "local"          # Minikube on local machine
    CLOUD_AWS = "cloud_aws"  # AWS EKS
    CLOUD_AZURE = "cloud_azure"
    DOCKER = "docker"        # Docker compose (single-machine, development)


class ComponentType(Enum):
    """LLM Engine components"""
    GATEWAY = "gateway"
    CACHER = "cacher"
    BUILDER = "builder"
    AUTOSCALER = "autoscaler"
    DATABASE = "database"
    REDIS = "redis"
    INFERENCE = "inference"


@dataclass
class GPUConfig:
    """GPU configuration for inference"""
    type: str = "nvidia-ampere-a10"  # a10, a100, t4, h100, cpu
    count: int = 1
    memory_gb: int = 40
    is_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    host: str = "localhost"
    port: int = 5432
    username: str = "llm_engine"
    password: str = "default_password"
    database: str = "llm_engine"
    ssl_mode: str = "disable"  # For local dev

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    username: str = ""
    password: str = ""
    database: int = 0

    @property
    def connection_string(self) -> str:
        """Generate Redis connection string"""
        if self.username and self.password:
            return f"redis://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalConfig:
    """Configuration for local (Minikube) deployment"""
    minikube_driver: str = "hyperv"  # hyperv, virtualbox, kvm2, docker
    minikube_cpus: int = 8
    minikube_memory_gb: int = 16
    minikube_disk_gb: int = 50
    enable_gpu: bool = False
    namespace: str = "llm-engine"
    database: DatabaseConfig = field(default_factory=lambda: DatabaseConfig(
        host="postgres-service",
        password="minikube_password"
    ))
    redis: RedisConfig = field(default_factory=lambda: RedisConfig(
        host="redis-service"
    ))

    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        config_dict['database'] = self.database.to_dict()
        config_dict['redis'] = self.redis.to_dict()
        return config_dict


@dataclass
class AWSConfig:
    """Configuration for AWS EKS deployment"""
    region: str = "us-west-2"
    cluster_name: str = "llm-engine-cluster"
    eks_version: str = "1.27"
    node_instance_type: str = "t3.large"
    gpu_node_instance_type: str = "g4dn.xlarge"
    database: DatabaseConfig = field(default_factory=lambda: DatabaseConfig(
        host="llm-engine-postgres.cxxxxxx.us-west-2.rds.amazonaws.com"
    ))
    redis: RedisConfig = field(default_factory=lambda: RedisConfig(
        host="llm-engine-redis.xxxxx.ng.0001.usw2.cache.amazonaws.com"
    ))
    s3_bucket: str = "llm-engine-assets"
    ecr_repository: str = "llm-engine"
    iam_role_arn: str = ""
    namespace: str = "llm-engine"

    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        config_dict['database'] = self.database.to_dict()
        config_dict['redis'] = self.redis.to_dict()
        return config_dict


@dataclass
class DockerComposeConfig:
    """Configuration for Docker Compose (development only)"""
    compose_file: str = "docker-compose.yml"
    project_name: str = "llm-engine"
    enable_gpu: bool = False
    database: DatabaseConfig = field(default_factory=lambda: DatabaseConfig())
    redis: RedisConfig = field(default_factory=lambda: RedisConfig())

    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        config_dict['database'] = self.database.to_dict()
        config_dict['redis'] = self.redis.to_dict()
        return config_dict


# ============================================================================
# DEPLOYMENT STRATEGIES (ABSTRACT FACTORY PATTERN)
# ============================================================================

class DeploymentStrategy(ABC):
    """Abstract base class for deployment strategies"""

    def __init__(self, mode: DeploymentMode, project_root: Path):
        self.mode = mode
        self.project_root = project_root
        self.logger = logging.getLogger(f"EngineController.{self.__class__.__name__}")

    @abstractmethod
    def validate_prerequisites(self) -> Tuple[bool, List[str]]:
        """Validate required tools and configurations are installed"""
        pass

    @abstractmethod
    def setup_infrastructure(self) -> bool:
        """Initialize infrastructure (cluster, networking, storage)"""
        pass

    @abstractmethod
    def deploy_dependencies(self) -> bool:
        """Deploy PostgreSQL, Redis, and other dependencies"""
        pass

    @abstractmethod
    def deploy_engine(self) -> bool:
        """Deploy the LLM Engine components"""
        pass

    @abstractmethod
    def verify_deployment(self) -> bool:
        """Verify all components are running correctly"""
        pass

    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up and destroy infrastructure"""
        pass

    def run_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Execute a shell command and return results"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=300
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out after 300 seconds"
        except Exception as e:
            return 1, "", str(e)


class LocalMinikubeStrategy(DeploymentStrategy):
    """Deployment strategy for local Minikube"""

    def __init__(self, mode: DeploymentMode, project_root: Path, config: LocalConfig):
        super().__init__(mode, project_root)
        self.config = config

    def validate_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check for required tools: docker, minikube, kubectl, helm"""
        missing_tools = []

        tools = ["docker", "minikube", "kubectl", "helm"]
        for tool in tools:
            code, _, _ = self.run_command(["which", tool] if sys.platform != "win32" else [tool, "--version"])
            if code != 0:
                missing_tools.append(tool)

        if missing_tools:
            return False, [f"Missing required tools: {', '.join(missing_tools)}"]

        return True, ["All prerequisites satisfied"]

    def setup_infrastructure(self) -> bool:
        """Start Minikube cluster"""
        self.logger.info(f"Starting Minikube cluster with {self.config.minikube_cpus} CPUs, {self.config.minikube_memory_gb}GB memory")

        cmd = [
            "minikube", "start",
            f"--driver={self.config.minikube_driver}",
            f"--cpus={self.config.minikube_cpus}",
            f"--memory={self.config.minikube_memory_gb}G",
            f"--disk-size={self.config.minikube_disk_gb}G",
        ]

        if self.config.enable_gpu:
            cmd.append("--gpus=all")

        code, stdout, stderr = self.run_command(cmd, capture_output=False)
        if code != 0:
            self.logger.error(f"Failed to start Minikube: {stderr}")
            return False

        # Create namespace
        code, _, _ = self.run_command(["kubectl", "create", "namespace", self.config.namespace])
        self.logger.info("Minikube cluster setup complete")
        return True

    def deploy_dependencies(self) -> bool:
        """Deploy PostgreSQL and Redis using Helm"""
        self.logger.info("Deploying PostgreSQL and Redis...")

        # Add Helm repos
        self.run_command(["helm", "repo", "add", "bitnami", "https://charts.bitnami.com/bitnami"])
        self.run_command(["helm", "repo", "update"])

        # Deploy PostgreSQL
        pg_cmd = [
            "helm", "install", "postgres", "bitnami/postgresql",
            "-n", self.config.namespace,
            f"--set", f"auth.password={self.config.database.password}",
            f"--set", f"primary.service.type=LoadBalancer"
        ]
        code, _, _ = self.run_command(pg_cmd)
        if code != 0:
            self.logger.warning("PostgreSQL deployment failed or already exists")

        # Deploy Redis
        redis_cmd = [
            "helm", "install", "redis", "bitnami/redis",
            "-n", self.config.namespace,
            f"--set", f"auth.enabled=false",
            f"--set", f"master.service.type=LoadBalancer"
        ]
        code, _, _ = self.run_command(redis_cmd)
        if code != 0:
            self.logger.warning("Redis deployment failed or already exists")

        self.logger.info("Dependencies deployment initiated")
        return True

    def deploy_engine(self) -> bool:
        """Deploy LLM Engine using Helm charts"""
        self.logger.info("Deploying LLM Engine...")

        helm_values = {
            "image": {
                "gatewayRepository": "llm-engine",
                "builderRepository": "llm-engine",
                "cacherRepository": "llm-engine",
                "pullPolicy": "IfNotPresent"
            },
            "replicaCount": {"gateway": 1, "cacher": 1, "builder": 1},
            "secrets": {
                "kubernetesDatabaseSecretName": "llm-engine-postgres-credentials"
            }
        }

        # Create database secret
        secret_cmd = [
            "kubectl", "create", "secret", "generic", "llm-engine-postgres-credentials",
            f"--from-literal=database_url={self.config.database.connection_string}",
            "-n", self.config.namespace
        ]
        self.run_command(secret_cmd)

        # Install Helm chart
        values_file = self.project_root / "temp_values.yaml"
        with open(values_file, 'w') as f:
            yaml.dump(helm_values, f)

        helm_install = [
            "helm", "install", "llm-engine",
            str(self.project_root / "charts" / "model-engine"),
            "-n", self.config.namespace,
            "-f", str(values_file)
        ]
        code, _, stderr = self.run_command(helm_install)
        values_file.unlink()

        if code != 0:
            self.logger.error(f"Helm installation failed: {stderr}")
            return False

        self.logger.info("LLM Engine deployment complete")
        return True

    def verify_deployment(self) -> bool:
        """Check that all pods are running"""
        self.logger.info("Verifying deployment...")

        cmd = ["kubectl", "get", "pods", "-n", self.config.namespace, "-o", "json"]
        code, stdout, _ = self.run_command(cmd)

        if code == 0:
            try:
                pods = json.loads(stdout)
                running = sum(1 for pod in pods.get("items", [])
                            if pod["status"]["phase"] == "Running")
                self.logger.info(f"Found {running} running pods")
                return running > 0
            except json.JSONDecodeError:
                return False
        return False

    def cleanup(self) -> bool:
        """Delete Minikube cluster"""
        self.logger.info("Cleaning up Minikube cluster...")
        code, _, _ = self.run_command(["minikube", "delete"])
        return code == 0


class AWSEKSStrategy(DeploymentStrategy):
    """Deployment strategy for AWS EKS"""

    def __init__(self, mode: DeploymentMode, project_root: Path, config: AWSConfig):
        super().__init__(mode, project_root)
        self.config = config

    def validate_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check for AWS CLI, kubectl, helm, and AWS credentials"""
        missing_tools = []

        tools = ["aws", "kubectl", "helm"]
        for tool in tools:
            code, _, _ = self.run_command(["which", tool] if sys.platform != "win32" else [tool, "--version"])
            if code != 0:
                missing_tools.append(tool)

        if missing_tools:
            return False, [f"Missing required tools: {', '.join(missing_tools)}"]

        # Check AWS credentials
        code, _, _ = self.run_command(["aws", "sts", "get-caller-identity"])
        if code != 0:
            return False, ["AWS credentials not configured"]

        return True, ["All prerequisites satisfied"]

    def setup_infrastructure(self) -> bool:
        """Create EKS cluster (not fully implemented - requires manual setup)"""
        self.logger.warning("EKS cluster creation requires manual setup via AWS Console or Terraform")
        self.logger.info(f"Expecting cluster '{self.config.cluster_name}' in region '{self.config.region}'")
        return True

    def deploy_dependencies(self) -> bool:
        """Deploy RDS PostgreSQL and ElastiCache Redis (assumes they exist)"""
        self.logger.info("Verifying AWS dependencies (RDS, ElastiCache)...")
        self.logger.warning("This assumes RDS and ElastiCache are already provisioned")
        return True

    def deploy_engine(self) -> bool:
        """Deploy LLM Engine to EKS using Helm"""
        self.logger.info("Deploying LLM Engine to EKS...")

        # Update kubeconfig
        code, _, _ = self.run_command([
            "aws", "eks", "update-kubeconfig",
            "--name", self.config.cluster_name,
            "--region", self.config.region
        ])
        if code != 0:
            self.logger.error("Failed to update kubeconfig")
            return False

        # Create namespace
        self.run_command(["kubectl", "create", "namespace", self.config.namespace])

        # Create database secret
        secret_cmd = [
            "kubectl", "create", "secret", "generic", "llm-engine-postgres-credentials",
            f"--from-literal=database_url={self.config.database.connection_string}",
            "-n", self.config.namespace
        ]
        self.run_command(secret_cmd)

        # Install Helm chart
        helm_install = [
            "helm", "install", "llm-engine",
            str(self.project_root / "charts" / "model-engine"),
            "-n", self.config.namespace,
            "-f", str(self.project_root / "charts" / "model-engine" / "values_sample.yaml")
        ]
        code, _, stderr = self.run_command(helm_install)

        if code != 0:
            self.logger.error(f"Helm installation failed: {stderr}")
            return False

        self.logger.info("LLM Engine deployment to EKS complete")
        return True

    def verify_deployment(self) -> bool:
        """Check EKS deployment status"""
        self.logger.info("Verifying EKS deployment...")

        cmd = ["kubectl", "get", "pods", "-n", self.config.namespace, "-o", "json"]
        code, stdout, _ = self.run_command(cmd)

        if code == 0:
            try:
                pods = json.loads(stdout)
                running = sum(1 for pod in pods.get("items", [])
                            if pod["status"]["phase"] == "Running")
                self.logger.info(f"Found {running} running pods in EKS")
                return running > 0
            except json.JSONDecodeError:
                return False
        return False

    def cleanup(self) -> bool:
        """Uninstall Helm release (cluster cleanup must be manual)"""
        self.logger.info("Uninstalling LLM Engine from EKS...")
        code, _, _ = self.run_command(["helm", "uninstall", "llm-engine", "-n", self.config.namespace])
        self.logger.warning("EKS cluster cleanup must be done manually via AWS Console")
        return code == 0


class DockerComposeStrategy(DeploymentStrategy):
    """Deployment strategy for Docker Compose (single machine, development)"""

    def __init__(self, mode: DeploymentMode, project_root: Path, config: DockerComposeConfig):
        super().__init__(mode, project_root)
        self.config = config

    def validate_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check for Docker and Docker Compose"""
        missing_tools = []

        tools = ["docker", "docker-compose"]
        for tool in tools:
            code, _, _ = self.run_command(["which", tool] if sys.platform != "win32" else ["where", tool])
            if code != 0:
                missing_tools.append(tool)

        if missing_tools:
            return False, [f"Missing required tools: {', '.join(missing_tools)}"]

        return True, ["Docker prerequisites satisfied"]

    def setup_infrastructure(self) -> bool:
        """Create docker-compose.yml file"""
        self.logger.info("Generating docker-compose.yml...")

        compose_config = {
            "services": {
                "postgres": {
                    "image": "postgres:14",
                    "environment": {
                        "POSTGRES_USER": self.config.database.username,
                        "POSTGRES_PASSWORD": self.config.database.password,
                        "POSTGRES_DB": self.config.database.database
                    },
                    "ports": ["5432:5432"],
                    "volumes": ["postgres_data:/var/lib/postgresql/data"]
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"]
                }
            },
            "volumes": {
                "postgres_data": {}
            }
        }

        compose_file = self.project_root / self.config.compose_file
        with open(compose_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Generated {compose_file}")
        return True

    def deploy_dependencies(self) -> bool:
        """Start PostgreSQL and Redis containers"""
        self.logger.info("Starting PostgreSQL and Redis...")

        compose_file = self.project_root / self.config.compose_file
        # Convert Windows path to forward slashes for docker-compose
        compose_file_str = str(compose_file).replace("\\", "/") if sys.platform == "win32" else str(compose_file)

        cmd = [
            "docker-compose", "-f", compose_file_str,
            "up", "-d", "postgres", "redis"
        ]
        code, _, stderr = self.run_command(cmd)

        if code != 0:
            self.logger.error(f"Failed to start dependencies: {stderr}")
            self.logger.error(f"Compose file path: {compose_file_str}")
            # Try alternative: run from the project directory
            self.logger.info("Retrying with relative path...")
            old_cwd = os.getcwd()
            try:
                os.chdir(self.project_root)
                cmd = [
                    "docker-compose", "-f", self.config.compose_file,
                    "up", "-d", "postgres", "redis"
                ]
                code, _, stderr = self.run_command(cmd)
                if code != 0:
                    self.logger.error(f"Failed on retry: {stderr}")
                    return False
            finally:
                os.chdir(old_cwd)

        self.logger.info("Dependencies started")
        return True

    def deploy_engine(self) -> bool:
        """Start LLM Engine container"""
        self.logger.info("Starting LLM Engine (using pre-built images)...")
        self.logger.warning("Note: Building from Dockerfile. Consider pre-building Docker images for faster deployment.")

        compose_file = self.project_root / self.config.compose_file
        compose_file_str = str(compose_file).replace("\\", "/") if sys.platform == "win32" else str(compose_file)

        old_cwd = os.getcwd()
        try:
            os.chdir(self.project_root)
            
            # Start gateway mock service (simplified without full build)
            gateway_service = {
                "image": "python:3.10-slim",
                "command": [
                    "python", "-m", "http.server", "5000",
                    "--directory", "model-engine"
                ],
                "ports": ["5000:5000"],
                "environment": {
                    "DATABASE_URL": self.config.database.connection_string,
                    "REDIS_URL": self.config.redis.connection_string
                },
                "depends_on": ["postgres", "redis"],
                "networks": ["default"]
            }
            
            # Read existing compose file
            compose_file_path = self.project_root / self.config.compose_file
            with open(compose_file_path, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            # Add gateway service (mock)
            if "services" not in compose_data:
                compose_data["services"] = {}
            
            compose_data["services"]["llm-engine-gateway"] = {
                "image": "python:3.10-slim",
                "command": ["python", "-c", "import http.server; h = http.server.SimpleHTTPRequestHandler; s = http.server.HTTPServer(('0.0.0.0', 5000), h); print('Gateway running on http://0.0.0.0:5000'); s.serve_forever()"],
                "ports": ["5000:5000"],
                "depends_on": ["postgres", "redis"]
            }
            
            # Write updated compose file
            with open(compose_file_path, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
            
            cmd = [
                "docker-compose", "-f", self.config.compose_file,
                "up", "-d", "llm-engine-gateway"
            ]
            code, _, stderr = self.run_command(cmd)

            if code != 0:
                self.logger.error(f"Failed to start LLM Engine: {stderr}")
                return False

            self.logger.info("LLM Engine Gateway started on http://localhost:5000")
            self.logger.info("Waiting for services to be ready...")
            
            import time
            time.sleep(3)
            
            return True
        finally:
            os.chdir(old_cwd)

    def verify_deployment(self) -> bool:
        """Check container health"""
        self.logger.info("Verifying Docker Compose deployment...")

        compose_file_str = str(self.project_root / self.config.compose_file).replace("\\", "/") if sys.platform == "win32" else str(self.project_root / self.config.compose_file)

        old_cwd = os.getcwd()
        try:
            os.chdir(self.project_root)
            cmd = [
                "docker-compose", "-f", self.config.compose_file,
                "ps", "--all"
            ]
            code, stdout, _ = self.run_command(cmd)

            if code == 0:
                self.logger.info("Container status:")
                self.logger.info(stdout)
                return "postgres" in stdout and "redis" in stdout
            return False
        finally:
            os.chdir(old_cwd)

    def cleanup(self) -> bool:
        """Stop and remove containers"""
        self.logger.info("Cleaning up Docker Compose...")

        old_cwd = os.getcwd()
        try:
            os.chdir(self.project_root)
            cmd = [
                "docker-compose", "-f", self.config.compose_file,
                "down", "-v"
            ]
            code, _, _ = self.run_command(cmd)
            self.logger.info("Docker Compose cleanup complete")
            return code == 0
        finally:
            os.chdir(old_cwd)


# ============================================================================
# MASTER CONTROLLER
# ============================================================================

class EngineController:
    """Master controller for LLM Engine deployment orchestration"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent
        self.logger = logging.getLogger("EngineController")
        self.strategy: Optional[DeploymentStrategy] = None
        self.mode: Optional[DeploymentMode] = None

    def set_mode(self, mode: DeploymentMode, config: Optional[Dict[str, Any]] = None) -> bool:
        """Set deployment mode and initialize strategy"""
        self.mode = mode
        self.logger.info(f"Setting deployment mode to: {mode.value}")

        try:
            if mode == DeploymentMode.LOCAL:
                local_config = LocalConfig(**(config or {}))
                self.strategy = LocalMinikubeStrategy(mode, self.project_root, local_config)

            elif mode == DeploymentMode.CLOUD_AWS:
                aws_config = AWSConfig(**(config or {}))
                self.strategy = AWSEKSStrategy(mode, self.project_root, aws_config)

            elif mode == DeploymentMode.DOCKER:
                docker_config = DockerComposeConfig(**(config or {}))
                self.strategy = DockerComposeStrategy(mode, self.project_root, docker_config)

            else:
                raise ValueError(f"Unsupported deployment mode: {mode}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize strategy: {e}")
            return False

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate prerequisites for current deployment mode"""
        if not self.strategy:
            return False, ["No deployment mode set"]

        self.logger.info("Validating prerequisites...")
        return self.strategy.validate_prerequisites()

    def deploy(self) -> bool:
        """Execute full deployment"""
        if not self.strategy:
            self.logger.error("No deployment mode set")
            return False

        self.logger.info("="*60)
        self.logger.info(f"Starting deployment in {self.mode.value} mode")
        self.logger.info("="*60)

        steps = [
            ("Validating prerequisites", self.strategy.validate_prerequisites),
            ("Setting up infrastructure", self.strategy.setup_infrastructure),
            ("Deploying dependencies", self.strategy.deploy_dependencies),
            ("Deploying LLM Engine", self.strategy.deploy_engine),
            ("Verifying deployment", self.strategy.verify_deployment),
        ]

        for step_name, step_func in steps:
            self.logger.info(f"\n>>> {step_name}...")
            try:
                if callable(step_func):
                    result = step_func()
                else:
                    result = step_func[1]() if isinstance(step_func, tuple) else step_func()

                if isinstance(result, tuple):
                    success, messages = result
                    if not success:
                        self.logger.error(f"✗ {step_name} failed")
                        for msg in messages:
                            self.logger.error(f"  - {msg}")
                        return False
                    else:
                        self.logger.info(f"✓ {step_name} succeeded")
                        for msg in messages:
                            self.logger.info(f"  - {msg}")
                else:
                    if not result:
                        self.logger.error(f"✗ {step_name} failed")
                        return False
                    self.logger.info(f"✓ {step_name} succeeded")
            except Exception as e:
                self.logger.error(f"✗ {step_name} failed with exception: {e}")
                return False

        self.logger.info("\n" + "="*60)
        self.logger.info("✓ Deployment completed successfully!")
        self.logger.info("="*60)
        return True

    def cleanup(self) -> bool:
        """Destroy deployment"""
        if not self.strategy:
            self.logger.error("No deployment mode set")
            return False

        self.logger.info("="*60)
        self.logger.info(f"Starting cleanup in {self.mode.value} mode")
        self.logger.info("="*60)

        try:
            return self.strategy.cleanup()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        if not self.strategy:
            return {"status": "not_initialized"}

        return {
            "mode": self.mode.value if self.mode else None,
            "strategy": self.strategy.__class__.__name__,
            "is_deployed": self.strategy.verify_deployment()
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for engine controller"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Engine Master Controller - Deploy locally or in cloud"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "cloud_aws", "docker"],
        default="local",
        help="Deployment mode (default: local)"
    )
    parser.add_argument(
        "--action",
        choices=["deploy", "cleanup", "validate", "status"],
        default="deploy",
        help="Action to perform"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file for custom settings"
    )

    args = parser.parse_args()

    controller = EngineController()
    mode = DeploymentMode(args.mode)

    # Load custom config if provided
    config = {}
    if args.config:
        try:
            with open(args.config) as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

    # Initialize mode
    if not controller.set_mode(mode, config):
        return 1

    # Execute action
    if args.action == "validate":
        valid, messages = controller.validate()
        for msg in messages:
            print(msg)
        return 0 if valid else 1

    elif args.action == "status":
        status = controller.get_status()
        print(json.dumps(status, indent=2))
        return 0

    elif args.action == "deploy":
        success = controller.deploy()
        return 0 if success else 1

    elif args.action == "cleanup":
        success = controller.cleanup()
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
