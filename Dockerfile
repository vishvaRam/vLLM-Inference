# Use your existing vLLM image as the base
FROM vishva123/vllm-server-cuda-12.8.1

# Set the working directory
WORKDIR /app

COPY Model/Qwen3-VL-8B-Instruct-W4A16-AutoRound /models/llm
