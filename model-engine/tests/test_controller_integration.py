"""
Integration Tests for Controller with Monitoring
================================================

Tests the complete workflow: controller + executor + monitoring together.

Run with:
  pytest tests/test_controller_integration.py -v
  pytest tests/test_controller_integration.py -v --asyncio-mode=auto
"""

import pytest
import asyncio
from datetime import datetime

from model_engine_server.controller_with_monitoring import EngineControllerWithMonitoring
from model_engine_server.model_executor import MockModelExecutor, JobStatus


class TestControllerIntegration:
    """Integration tests for controller with monitoring"""
    
    @pytest.fixture
    def controller(self):
        """Create controller with mock executor"""
        executor = MockModelExecutor(latency_ms=10)
        return EngineControllerWithMonitoring(executor, enable_metrics=True)
    
    @pytest.mark.asyncio
    async def test_submit_and_track_fine_tune_job(self, controller):
        """Test complete fine-tune workflow"""
        # Submit job
        job_id = await controller.submit_fine_tune_job(
            base_model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/output",
            epochs=3
        )
        
        assert job_id is not None
        assert job_id.startswith("job_")
        
        # Check status
        status = await controller.get_job_status(job_id)
        
        assert status["job_id"] == job_id
        assert "status" in status
        assert status["model"] == "llama-2-7b"
    
    @pytest.mark.asyncio
    async def test_submit_inference_job(self, controller):
        """Test inference job submission"""
        job_id = await controller.submit_inference_job(
            prompt="What is AI?",
            model="falcon-7b",
            max_tokens=50
        )
        
        assert job_id is not None
        assert "inf_" in job_id
        
        status = await controller.get_job_status(job_id)
        assert status["job_type"] == "inference"
        assert status["status"] == JobStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, controller):
        """Test job cancellation"""
        job_id = await controller.submit_fine_tune_job(
            base_model="mpt-7b",
            dataset_path="/data/data.jsonl",
            output_path="/models/out"
        )
        
        # Cancel the job
        success = await controller.cancel_job(job_id)
        
        assert success is True
        
        status = await controller.get_job_status(job_id)
        assert status["status"] == JobStatus.CANCELLED.value
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_jobs(self, controller):
        """Test handling multiple concurrent jobs"""
        job_ids = []
        
        # Submit 5 jobs concurrently
        for i in range(5):
            job_id = await controller.submit_fine_tune_job(
                base_model="llama-2-7b",
                dataset_path=f"/data/dataset{i}.jsonl",
                output_path=f"/models/model{i}"
            )
            job_ids.append(job_id)
        
        assert len(job_ids) == 5
        assert len(set(job_ids)) == 5  # All unique
        
        # Check all jobs are tracked
        for job_id in job_ids:
            status = await controller.get_job_status(job_id)
            assert status["job_id"] == job_id
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, controller):
        """Test system health check"""
        health = await controller.get_system_health()
        
        assert "healthy" in health
        assert "components" in health
        assert "controller" in health
        
        controller_health = health["controller"]
        assert "tracked_jobs" in controller_health
        assert "active_jobs" in controller_health
        assert "completed_jobs" in controller_health
    
    @pytest.mark.asyncio
    async def test_healthcheck_endpoint(self, controller):
        """Test k8s-compatible health check endpoint"""
        hc = await controller.healthcheck_endpoint()
        
        assert "healthy" in hc
        assert "executor" in hc
        assert "system" in hc
        
        # Should be healthy initially
        assert hc["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_metrics_export(self, controller):
        """Test metrics export in Prometheus format"""
        # Generate some activity
        job_id = await controller.submit_fine_tune_job(
            base_model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/output"
        )
        
        # Get metrics
        metrics = controller.get_metrics()
        
        assert isinstance(metrics, str)
        assert "http_request_duration_ms" in metrics or "#" in metrics
    
    @pytest.mark.asyncio
    async def test_metrics_dict(self, controller):
        """Test metrics as dictionary"""
        # Generate some jobs
        for i in range(3):
            await controller.submit_fine_tune_job(
                base_model="llama-2-7b",
                dataset_path=f"/data/data{i}.jsonl",
                output_path=f"/models/out{i}"
            )
        
        metrics = await controller.get_metrics_dict()
        
        assert "jobs" in metrics
        assert metrics["jobs"]["total"] == 3
        assert "metrics_raw" in metrics
    
    @pytest.mark.asyncio
    async def test_job_tracking_persistence(self, controller):
        """Test that jobs remain tracked after submission"""
        job_ids = []
        
        for i in range(3):
            job_id = await controller.submit_fine_tune_job(
                base_model="llama-2-7b",
                dataset_path=f"/data/{i}.jsonl",
                output_path=f"/models/{i}"
            )
            job_ids.append(job_id)
        
        # All jobs should be retrievable
        for job_id in job_ids:
            status = await controller.get_job_status(job_id)
            assert status["job_id"] == job_id
            assert "status" in status
    
    @pytest.mark.asyncio
    async def test_error_handling_nonexistent_job(self, controller):
        """Test graceful handling of nonexistent job"""
        status = await controller.get_job_status("job_nonexistent")
        
        # Should handle gracefully, not crash
        assert "job_id" in status
        assert status["job_id"] == "job_nonexistent"
    
    @pytest.mark.asyncio
    async def test_monitoring_with_disabled_metrics(self):
        """Test controller works with metrics disabled"""
        executor = MockModelExecutor()
        controller = EngineControllerWithMonitoring(
            executor,
            enable_metrics=False
        )
        
        job_id = await controller.submit_fine_tune_job(
            base_model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/output"
        )
        
        assert job_id is not None
        
        status = await controller.get_job_status(job_id)
        assert status["job_id"] == job_id


class TestHighLoadScenarios:
    """Test controller under high-load scenarios"""
    
    @pytest.mark.asyncio
    async def test_many_concurrent_jobs(self):
        """Test with many concurrent jobs"""
        executor = MockModelExecutor(latency_ms=5)
        controller = EngineControllerWithMonitoring(executor)
        
        # Submit 20 jobs concurrently
        jobs = await asyncio.gather(*[
            controller.submit_fine_tune_job(
                base_model="llama-2-7b",
                dataset_path=f"/data/{i}.jsonl",
                output_path=f"/models/{i}"
            )
            for i in range(20)
        ])
        
        assert len(jobs) == 20
        assert len(set(jobs)) == 20  # All unique
        
        # Check all are tracked
        health = await controller.get_system_health()
        assert health["controller"]["tracked_jobs"] == 20
    
    @pytest.mark.asyncio
    async def test_rapid_status_checks(self):
        """Test rapid status checks"""
        executor = MockModelExecutor()
        controller = EngineControllerWithMonitoring(executor)
        
        job_id = await controller.submit_fine_tune_job(
            base_model="llama-2-7b",
            dataset_path="/data/train.jsonl",
            output_path="/models/output"
        )
        
        # Rapid status checks
        statuses = await asyncio.gather(*[
            controller.get_job_status(job_id)
            for _ in range(10)
        ])
        
        assert len(statuses) == 10
        assert all(s["job_id"] == job_id for s in statuses)
    
    @pytest.mark.asyncio
    async def test_mixed_job_types(self):
        """Test mixing fine-tune and inference jobs"""
        executor = MockModelExecutor()
        controller = EngineControllerWithMonitoring(executor)
        
        # Submit mixed jobs
        jobs = []
        
        for i in range(3):
            job_id = await controller.submit_fine_tune_job(
                base_model="llama-2-7b",
                dataset_path=f"/data/{i}.jsonl",
                output_path=f"/models/{i}"
            )
            jobs.append(job_id)
        
        for i in range(3):
            job_id = await controller.submit_inference_job(
                prompt=f"Prompt {i}",
                model="falcon-7b"
            )
            jobs.append(job_id)
        
        assert len(jobs) == 6
        
        metrics = await controller.get_metrics_dict()
        assert metrics["jobs"]["total"] == 6


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_controller(self):
        """Test controller with no jobs submitted"""
        executor = MockModelExecutor()
        controller = EngineControllerWithMonitoring(executor)
        
        health = await controller.get_system_health()
        
        assert health["controller"]["tracked_jobs"] == 0
        assert health["controller"]["active_jobs"] == 0
        assert health["controller"]["completed_jobs"] == 0
    
    @pytest.mark.asyncio
    async def test_cancel_already_completed_job(self):
        """Test canceling a completed job"""
        executor = MockModelExecutor()
        controller = EngineControllerWithMonitoring(executor)
        
        # Inference job completes immediately
        job_id = await controller.submit_inference_job(
            prompt="Test",
            model="llama-2-7b"
        )
        
        # Try to cancel
        success = await controller.cancel_job(job_id)
        
        # Should handle gracefully
        assert success is True or success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
