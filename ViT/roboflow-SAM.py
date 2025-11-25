"""Roboflow Single Image Test.

This script runs the Roboflow workflow on a single image for testing purposes.
"""

import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ROBOFLOW_INFERENCE_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_INFERENCE_API_KEY not found in environment variables")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Replace 'YOUR_IMAGE.jpg' with the path to your image
result = client.run_workflow(
    workspace_name="dontloseyourheadsu",
    workflow_id="find-white-dots-and-thin-black-lines",
    images={
        "image": "YOUR_IMAGE.jpg"
    },
    use_cache=True # cache workflow definition for 15 minutes
)
