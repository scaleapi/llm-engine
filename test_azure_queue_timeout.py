#!/usr/bin/env python3
"""
Integration test script for Azure Service Bus queue timeout duration.
This script tests the actual Azure Service Bus queue creation with custom lock duration.

Usage:
    export SERVICEBUS_NAMESPACE="your-namespace"
    export AZURE_TENANT_ID="your-tenant-id"
    export AZURE_CLIENT_ID="your-client-id"
    export AZURE_CLIENT_SECRET="your-client-secret"
    python test_azure_queue_timeout.py
"""

import asyncio
import os
import sys
from datetime import timedelta
from azure.servicebus.management import ServiceBusAdministrationClient, QueueProperties
from azure.identity import DefaultAzureCredential

async def test_queue_creation_with_timeout():
    """Test creating Azure Service Bus queue with custom lock duration"""
    
    # Check required environment variables
    namespace = os.getenv("SERVICEBUS_NAMESPACE")
    if not namespace:
        print("ERROR: SERVICEBUS_NAMESPACE environment variable not set")
        return False
    
    try:
        # Create Service Bus Administration Client
        client = ServiceBusAdministrationClient(
            f"{namespace}.servicebus.windows.net",
            credential=DefaultAzureCredential()
        )
        
        test_cases = [
            {"timeout": 60, "description": "default timeout (60s)"},
            {"timeout": 120, "description": "custom timeout (120s)"},
            {"timeout": 300, "description": "maximum timeout (300s)"},
        ]
        
        for i, test_case in enumerate(test_cases):
            queue_name = f"test-queue-timeout-{i}-{test_case['timeout']}s"
            timeout_seconds = test_case["timeout"]
            
            print(f"\n--- Testing {test_case['description']} ---")
            
            try:
                # Create queue with custom lock duration
                queue_properties = QueueProperties(
                    lock_duration=timedelta(seconds=timeout_seconds)
                )
                
                print(f"Creating queue: {queue_name}")
                client.create_queue(queue_name, queue_properties=queue_properties)
                print(f"✓ Queue created successfully")
                
                # Verify the queue properties
                queue_props = client.get_queue(queue_name)
                actual_lock_duration = queue_props.lock_duration.total_seconds()
                
                print(f"Expected lock duration: {timeout_seconds}s")
                print(f"Actual lock duration: {actual_lock_duration}s")
                
                if actual_lock_duration == timeout_seconds:
                    print(f"✓ Lock duration matches expected value")
                else:
                    print(f"✗ Lock duration mismatch!")
                    return False
                
                # Clean up - delete the test queue
                client.delete_queue(queue_name)
                print(f"✓ Test queue deleted")
                
            except Exception as e:
                print(f"✗ Error testing {test_case['description']}: {e}")
                return False
        
        # Test validation error for timeout > 300s
        print(f"\n--- Testing validation error for timeout > 300s ---")
        try:
            queue_properties = QueueProperties(
                lock_duration=timedelta(seconds=400)  # Should fail
            )
            client.create_queue("test-invalid-timeout", queue_properties=queue_properties)
            print("✗ Should have failed for timeout > 300s")
            return False
        except Exception as e:
            print(f"✓ Correctly rejected timeout > 300s: {e}")
        
        print(f"\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Connection or authentication error: {e}")
        print("Make sure you have the correct Azure credentials set up")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_queue_creation_with_timeout())
    sys.exit(0 if success else 1)
