# Navigate to a directory with space (like /mnt/data if allowed, or your home)
cd /home/radixark/dbyang

# Pull the SGLang Docker image
docker pull lmsysorg/sglang:latest

# Run Docker with:
# - GPU access
# - Your code mounted
# - HuggingFace cache mounted from the large disk
# - Your branch code mounted
docker run --gpus all -it --rm \
  --shm-size 32g \
  -v /home/radixark/dbyang/sglang:/workspace/sglang \
  -v /home/radixark/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  bash

for disk space b200:

docker run --gpus all -it --rm \
  --shm-size 32g \
  -v /home/radixark/dbyang/sglang:/workspace/sglang \
  -v /home/radixark/.cache/huggingface:/root/.cache/huggingface \
  -v /data/model/dbyang/tmp:/tmp \
  -e HF_HOME=/root/.cache/huggingface \
  -e TMPDIR=/tmp \
  lmsysorg/sglang:latest \
  bash



# fetch branch
git fetch origin qwen-benchmarking:qwen-benchmarking

# Install python deps
pip install -e "python[all]"
# or from mount
pip install -e "python[all]" --no-deps
pip install sgl-kernel==0.3.20 --no-cache-dir

# run test
python test/registered/8-gpu-models/test_qwen3_235b_fp8.py
