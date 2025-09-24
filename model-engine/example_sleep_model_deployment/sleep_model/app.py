"""
Simple sleep model for testing queue timeout duration.
This model sleeps for 70 seconds to test queue lock duration > 60 seconds.
"""

import time
from typing import Any, Dict
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that sleeps for 70 seconds to test queue timeout.
    """
    try:
        data = request.get_json()
        sleep_duration = data.get('sleep_duration', 70)  # Default 70 seconds
        
        print(f"Starting inference... will sleep for {sleep_duration} seconds")
        
        # Sleep to simulate long-running inference
        time.sleep(sleep_duration)
        
        response = {
            "result": f"Completed after sleeping for {sleep_duration} seconds",
            "input": data,
            "status": "success"
        }
        
        print(f"Inference completed successfully after {sleep_duration} seconds")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/readyz', methods=['GET'])
def ready():
    """Readiness check endpoint"""
    return jsonify({"status": "ready"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
