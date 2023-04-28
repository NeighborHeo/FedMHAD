#!/bin/bash
set -e

# Set the script directory as the working directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Set the script directory to the Python path
SCRIPT_DIR="$(pwd)"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR"

# Generate a random experiment key (characters and numbers only 48 characters long)
experiment_key=$(openssl rand -base64 48 | sed 's/[^a-zA-Z0-9]//g')
echo $experiment_key

port=$(python -c "import random; print(random.randint(10000, 65535))")
echo $port

GPU_ID=0
# Download the CIFAR-10 dataset
# python -c "from torchvision.datasets import CIFAR10; CIFAR10('~/.data', download=True)"

# Download the EfficientNetB0 model
python -c "import torch; torch.hub.load( 'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)"

echo "Starting server"
CUDA_VISIBLE_DEVICES=$GPU_ID python server.py \
    --experiment_key "$experiment_key" \
    --port "$port" &
sleep 10  # Sleep for 3s to give the server enough time to start

for i in `seq 0 1`; do
    echo "Starting client $i"
    CUDA_VISIBLE_DEVICES=$GPU_ID python client.py \
        --index "$i" \
        --experiment_key "$experiment_key" \
        --port "$port" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

