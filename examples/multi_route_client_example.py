#!/usr/bin/env python3
"""
Example demonstrating how to deploy a multi-route FastAPI server using Launch.

This example shows how to use the new route configuration parameters to deploy
a FastAPI server with multiple endpoints that can be accessed through their
natural paths rather than being restricted to just /predict.
"""

from llmengine import Model
from llmengine.data_types.model_endpoints import CreateLLMEndpointRequest
from llmengine.data_types.core import ModelEndpointType
import requests
import time


def create_multi_route_endpoint():
    """
    Create a model endpoint with multiple routes using the new passthrough forwarder.
    """

    # Define the routes we want to expose from our FastAPI server
    custom_routes = [
        "/v1/chat/completions",  # OpenAI-compatible chat endpoint
        "/v1/completions",  # OpenAI-compatible completions endpoint
        "/analyze",  # Custom analysis endpoint
        "/custom/endpoint",  # Custom GET endpoint
        "/batch/process",  # Batch processing endpoint
    ]

    print("Creating model endpoint with multiple routes...")
    print(f"Routes to be exposed: {custom_routes}")

    # Create the endpoint with multi-route support
    response = Model.create(
        name="multi-route-fastapi-example",
        model="llama-2-7b",  # This is just for the bundle creation, our custom server will handle the logic
        inference_framework_image_tag="latest",
        # Hardware configuration
        cpus=4,
        memory="8Gi",
        storage="20Gi",
        gpus=1,
        gpu_type="nvidia-ampere-a10",
        # Scaling configuration
        min_workers=1,
        max_workers=3,
        per_worker=10,
        endpoint_type=ModelEndpointType.STREAMING,
        # NEW: Multi-route configuration
        routes=custom_routes,  # List of routes to forward
        forwarder_type="passthrough",  # Enable passthrough forwarding
        # Other settings
        public_inference=False,
        labels={"example": "multi-route", "type": "fastapi"},
    )

    print(f"Endpoint created! Task ID: {response.endpoint_creation_task_id}")
    return response.endpoint_creation_task_id


def test_multi_route_endpoint(endpoint_name: str, base_url: str):
    """
    Test the multi-route endpoint by making requests to different routes.
    """
    print(f"\nTesting multi-route endpoint: {endpoint_name}")
    print(f"Base URL: {base_url}")

    # Test cases for different routes
    test_cases = [
        {
            "name": "Traditional Predict",
            "method": "POST",
            "url": f"{base_url}/predict",
            "data": {"text": "Hello world", "model": "custom"},
        },
        {
            "name": "OpenAI Chat Completions",
            "method": "POST",
            "url": f"{base_url}/v1/chat/completions",
            "data": {
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "gpt-3.5-turbo",
                "max_tokens": 50,
            },
        },
        {
            "name": "OpenAI Completions",
            "method": "POST",
            "url": f"{base_url}/v1/completions",
            "data": {
                "prompt": "The future of AI is",
                "model": "text-davinci-003",
                "max_tokens": 50,
            },
        },
        {
            "name": "Custom Analysis",
            "method": "POST",
            "url": f"{base_url}/analyze",
            "data": {"text": "This is a good example of multi-route functionality"},
        },
        {
            "name": "Custom GET Endpoint",
            "method": "GET",
            "url": f"{base_url}/custom/endpoint",
            "data": None,
        },
        {
            "name": "Batch Processing",
            "method": "POST",
            "url": f"{base_url}/batch/process",
            "data": {"texts": ["First text", "Second text", "Third text"]},
        },
    ]

    # Execute test cases
    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")
        print(f"URL: {test_case['url']}")

        try:
            if test_case["method"] == "GET":
                response = requests.get(test_case["url"])
            else:
                response = requests.post(test_case["url"], json=test_case["data"])

            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result}")
            else:
                print(f"Error: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")


def main():
    """
    Main example workflow.
    """

    print("=" * 60)
    print("Launch Multi-Route FastAPI Server Example")
    print("=" * 60)

    print(
        """\
This example demonstrates the new multi-route passthrough functionality in Launch.

Instead of being limited to a single /predict endpoint, you can now:
1. Specify multiple routes to be forwarded to your FastAPI server
2. Use the passthrough forwarder type to enable full HTTP method support
3. Access your endpoints through their natural paths

Key benefits:
- No more single endpoint limitation
- Full FastAPI server compatibility
- Support for GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
- OpenAI-compatible endpoints alongside custom routes
- Easy migration of existing FastAPI applications
"""
    )

    # Step 1: Create the multi-route endpoint
    task_id = create_multi_route_endpoint()

    print(f"\nEndpoint creation initiated with task ID: {task_id}")
    print("Waiting for endpoint to be ready...")

    # In a real scenario, you would poll the endpoint status
    # For this example, we'll simulate waiting
    print("⏳ Endpoint is being deployed...")
    print("⏳ This may take several minutes...")

    # Step 2: Once ready, test the endpoints
    # Note: In practice, you'd get the actual endpoint URL from the Launch API
    endpoint_name = "multi-route-fastapi-example"
    base_url = f"https://your-launch-domain.com/v1/endpoints/{endpoint_name}"

    print(f"\n✅ Endpoint ready! You can now test it at: {base_url}")
    print("\nExample test calls you can make:")

    # Show example curl commands
    curl_examples = [
        {
            "name": "Traditional predict",
            "cmd": f'curl -X POST {base_url}/predict -H "Content-Type: application/json" -d \'{{"text": "Hello world", "model": "custom"}}\'',
        },
        {
            "name": "OpenAI chat",
            "cmd": f'curl -X POST {base_url}/v1/chat/completions -H "Content-Type: application/json" -d \'{{"messages": [{{"role": "user", "content": "Hello!"}}], "model": "gpt-3.5-turbo"}}\'',
        },
        {
            "name": "Custom analysis",
            "cmd": f'curl -X POST {base_url}/analyze -H "Content-Type: application/json" -d \'{{"text": "This is amazing!"}}\'',
        },
        {"name": "Custom GET endpoint", "cmd": f"curl -X GET {base_url}/custom/endpoint"},
    ]

    for example in curl_examples:
        print(f"\n{example['name']}:")
        print(f"  {example['cmd']}")

    print(f"\n" + "=" * 60)
    print("Multi-Route Support Successfully Configured!")
    print("=" * 60)

    # Uncomment the following line to run actual tests if you have a deployed endpoint
    # test_multi_route_endpoint(endpoint_name, base_url)


if __name__ == "__main__":
    main()
