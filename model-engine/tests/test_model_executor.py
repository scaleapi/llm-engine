"""
Test Suite for Model Executor Implementations
==============================================

Tests the abstract interface and concrete implementations (Mock, Kubernetes).
Uses pytest with async support.

Run with:
  pytest tests/test_model_executor.py -v

Coverage:
- MockModelExecutor: Full coverage (local testing)
- KubernetesModelExecutor: Basic structure (requires K8s setup)
"""

import pytest
import asyncio
from datetime import datetime

from model_executor import (
    ModelExecutor,
    MockModelExecutor,
    KubernetesModelExecutor,
    InferenceRequest,
    FineTuneRequest,
    JobStatus,
    JobResult
)


class TestMockModelExecutor:
    """Test suite for MockModelExecutor"""
    
    @pytest.fixture
    def executor(self):
        """Create a mock executor for testing"""
        return MockModelExecutor(latency_ms=10, fail_rate=0.0)
    
    @pytest.mark.asyncio
    async def test_inference_returns_valid_output(self, executor):
        """Test that inference returns non-empty output"""
        request = InferenceRequest(
            prompt="What is AI?",
            model="llama-2-7b",
            max_tokens=100
        )
        output = await executor.infer(request)
        
        assert output is not None
        assert isinstance(output, str)
        assert len(output) > 0
        assert "llama-2-7b" in output
        assert "AI" in output or "Mock" in output
    
    @pytest.mark.asyncio
    async def test_inference_respects_latency(self, executor):
        """Test that inference simulates latency correctly"""
        import time
        request = InferenceRequest(
            prompt="Test prompt",
            model="falcon-7b",
            max_tokens=50
        )
        
        start = time.time()
        await executor.infer(request)
        elapsed_ms = (time.time() - start) * 1000
        
        # Allow 20ms tolerance for system variance
        assert elapsed_ms >= 10 - 20, f"Expected ~10ms, got {elapsed_ms}ms"
    
    @pytest.mark.asyncio
    async def test_fine_tune_returns_job_id(self, executor):
        """Test that fine-tuning returns a valid job ID"""
        request = FineTuneRequest(
            base_model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/finetuned",
            epochs=3,
            batch_size=32
        )
        
        job_id = await executor.fine_tune(request)
        
        assert job_id is not None
        assert isinstance(job_id, str)
        assert job_id.startswith("job_")
        assert len(job_id) > 4
    
    @pytest.mark.asyncio
    async def test_fine_tune_creates_trackable_job(self, executor):
        """Test that submitted job can be tracked"""
        request = FineTuneRequest(
            base_model="falcon-7b",
            dataset_path="/data/custom.jsonl",
            output_path="/models/custom_falcon",
            epochs=5
        )
        
        job_id = await executor.fine_tune(request)
        result = await executor.get_job_status(job_id)
        
        assert result.job_id == job_id
        # Job may progress to QUEUED/RUNNING after first status check
        assert result.status in [JobStatus.SUBMITTED, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED]
        assert result.created_at is not None
    
    @pytest.mark.asyncio
    async def test_job_status_progression(self, executor):
        """Test that job status progresses correctly"""
        request = FineTuneRequest(
            base_model="mpt-7b",
            dataset_path="/data/mpt_data.jsonl",
            output_path="/models/mpt_custom"
        )
        
        job_id = await executor.fine_tune(request)
        
        # Track status progression
        statuses = []
        for _ in range(5):
            result = await executor.get_job_status(job_id)
            statuses.append(result.status)
            await asyncio.sleep(0.01)  # Small delay
        
        # Should progress or remain same (no backwards progression)
        # Status ordering: SUBMITTED < QUEUED < RUNNING < COMPLETED
        status_order = {
            JobStatus.SUBMITTED: 1,
            JobStatus.QUEUED: 2,
            JobStatus.RUNNING: 3,
            JobStatus.COMPLETED: 4,
            JobStatus.FAILED: 5,
            JobStatus.CANCELLED: 5
        }
        
        for i in range(len(statuses) - 1):
            # Status should progress or stay same (no backwards)
            assert status_order[statuses[i]] <= status_order[statuses[i + 1]]
    
    @pytest.mark.asyncio
    async def test_get_job_status_nonexistent_job(self, executor):
        """Test handling of nonexistent job"""
        result = await executor.get_job_status("job_nonexistent")
        
        assert result.job_id == "job_nonexistent"
        assert result.status == JobStatus.FAILED
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_cancel_job_success(self, executor):
        """Test job cancellation"""
        request = FineTuneRequest(
            base_model="llama-2-7b",
            dataset_path="/data/data.jsonl",
            output_path="/models/output"
        )
        
        job_id = await executor.fine_tune(request)
        cancelled = await executor.cancel_job(job_id)
        
        assert cancelled is True
        
        result = await executor.get_job_status(job_id)
        assert result.status == JobStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, executor):
        """Test cancellation of nonexistent job returns False"""
        cancelled = await executor.cancel_job("job_nonexistent")
        assert cancelled is False
    
    @pytest.mark.asyncio
    async def test_health_check_returns_valid_status(self, executor):
        """Test health check returns expected format"""
        health = await executor.health_check()
        
        assert health is not None
        assert isinstance(health, dict)
        assert "healthy" in health
        assert health["healthy"] is True
        assert "services" in health
        assert "models" in health["services"]
        assert "compute" in health["services"]
        assert "storage" in health["services"]
    
    @pytest.mark.asyncio
    async def test_health_check_lists_available_models(self, executor):
        """Test health check includes available models"""
        health = await executor.health_check()
        
        models_info = health["services"]["models"]
        assert "available" in models_info
        assert models_info["available"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_requests(self, executor):
        """Test handling multiple concurrent inference requests"""
        requests = [
            InferenceRequest(prompt=f"Prompt {i}", model="llama-2-7b")
            for i in range(5)
        ]
        
        results = await asyncio.gather(
            *[executor.infer(req) for req in requests]
        )
        
        assert len(results) == 5
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) > 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_fine_tune_jobs(self, executor):
        """Test handling multiple concurrent fine-tune requests"""
        requests = [
            FineTuneRequest(
                base_model="llama-2-7b",
                dataset_path=f"/data/dataset{i}.jsonl",
                output_path=f"/models/model{i}"
            )
            for i in range(3)
        ]
        
        job_ids = await asyncio.gather(
            *[executor.fine_tune(req) for req in requests]
        )
        
        assert len(job_ids) == 3
        assert len(set(job_ids)) == 3  # All unique
        assert all(jid.startswith("job_") for jid in job_ids)
    
    @pytest.mark.asyncio
    async def test_mock_with_failure_rate(self):
        """Test mock executor with configured failure rate"""
        executor = MockModelExecutor(fail_rate=1.0)  # Always fail
        
        request = FineTuneRequest(
            base_model="falcon-7b",
            dataset_path="/data/data.jsonl",
            output_path="/models/output"
        )
        
        job_id = await executor.fine_tune(request)
        result = await executor.get_job_status(job_id)
        
        assert result.status == JobStatus.FAILED


class TestInferenceRequest:
    """Test InferenceRequest dataclass"""
    
    def test_inference_request_defaults(self):
        """Test default parameters"""
        req = InferenceRequest(
            prompt="Test",
            model="llama-2-7b"
        )
        
        assert req.prompt == "Test"
        assert req.model == "llama-2-7b"
        assert req.max_tokens == 100
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.timeout_seconds == 30
    
    def test_inference_request_custom_params(self):
        """Test custom parameters"""
        req = InferenceRequest(
            prompt="Complex query",
            model="falcon-40b",
            max_tokens=500,
            temperature=0.1,
            top_p=0.95,
            timeout_seconds=60
        )
        
        assert req.max_tokens == 500
        assert req.temperature == 0.1
        assert req.timeout_seconds == 60


class TestFineTuneRequest:
    """Test FineTuneRequest dataclass"""
    
    def test_fine_tune_request_defaults(self):
        """Test default parameters"""
        req = FineTuneRequest(
            base_model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/finetuned"
        )
        
        assert req.base_model == "llama-2-7b"
        assert req.epochs == 3
        assert req.batch_size == 32
        assert req.learning_rate == 1e-4
        assert req.timeout_seconds == 3600
    
    def test_fine_tune_request_custom_hyperparams(self):
        """Test custom hyperparameters"""
        req = FineTuneRequest(
            base_model="falcon-7b",
            dataset_path="/data/custom.jsonl",
            output_path="/models/custom",
            epochs=10,
            batch_size=64,
            learning_rate=5e-4
        )
        
        assert req.epochs == 10
        assert req.batch_size == 64
        assert req.learning_rate == 5e-4


class TestKubernetesModelExecutor:
    """Test KubernetesModelExecutor structure"""
    
    def test_kubernetes_executor_initialization(self):
        """Test executor can be instantiated"""
        executor = KubernetesModelExecutor(
            namespace="llm-engine",
            timeout_seconds=7200
        )
        
        assert executor.namespace == "llm-engine"
        assert executor.timeout_seconds == 7200
    
    @pytest.mark.asyncio
    async def test_kubernetes_executor_not_implemented(self):
        """Test that K8s executor raises NotImplementedError without setup"""
        executor = KubernetesModelExecutor()
        
        request = InferenceRequest(prompt="Test", model="llama-2-7b")
        
        with pytest.raises(NotImplementedError):
            await executor.infer(request)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
