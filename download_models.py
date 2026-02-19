import subprocess
import os

model_id = "Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound"
local_dir = "./Model/Qwen3-VL-8B-Instruct-W4A16-AutoRound"

os.makedirs(local_dir, exist_ok=True)

cmd = [
    "huggingface-cli", "download",
    model_id,
    "--local-dir", local_dir,
    "--local-dir-use-symlinks", "False"
]

subprocess.run(cmd, check=True)
