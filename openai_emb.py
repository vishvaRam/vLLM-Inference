"""
Multimodal Embedding Demo with vLLM
-----------------------------------
This script demonstrates how to generate visual embeddings from images using
a multimodal model hosted on vLLM.

Usage:
1. Host your model: vllm serve Qwen/Qwen3-VL-Embedding-2B --task embed
2. Update the base_url with your server's IP.
"""

import base64
import io
from PIL import Image
from openai import OpenAI

# 1. Configuration
VLLM_SERVER_URL = "http://10.0.0.1:8000/v1"  # Replace with your vLLM IP
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
LOCAL_IMAGE_PATH = "assets/sample_document.png"


def encode_image_to_base64(image_path: str) -> str:
    """Helper function to convert local image to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (e.g., for RGBA PNGs)
            if img.mode != "RGB":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""


def get_multimodal_embedding():
    # 2. Initialize OpenAI Client
    client = OpenAI(api_key="EMPTY", base_url=VLLM_SERVER_URL)

    # 3. Prepare Image Data
    base64_image = encode_image_to_base64(LOCAL_IMAGE_PATH)
    if not base64_image:
        return

    # 4. Generate Embedding
    # Note: We use 'extra_body' to pass 'messages' because standard
    # OpenAI SDK validation doesn't natively support multimodal for embeddings.
    try:
        response = client.embeddings.create(
            model=MODEL_NAME,
            input="",  # Required by SDK, but ignored by vLLM when 'messages' is provided
            dimensions=1024,
            extra_body={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            {
                                "type": "text",
                                "text": "Represent this image for retrieval.",
                            },
                        ],
                    }
                ]
            },
            encoding_format="float",
        )

        embedding = response.data[0].embedding
        print(f"✅ Success! Generated embedding for: {LOCAL_IMAGE_PATH}")
        print(f"📏 Embedding Dimension: {len(embedding)}")
        print(f"🔢 First 5 values: {embedding[:5]}")

    except Exception as e:
        print(f"❌ Failed to get embedding: {e}")


if __name__ == "__main__":
    get_multimodal_embedding()
