#!/bin/bash

source env_setup.sh # Set up environment variables for Hugging Face cache and others

# Default values
MODEL_ID="Qwen/Qwen3-8B"
DATASET_PATH="data/passport_grpo.jsonl"
OUTPUT_DIR="Qwen3-8B-GRPO"
LEARNING_RATE=1e-5
EPOCHS=2
BATCH_SIZE=16
METHOD="transformers"  # Default training method

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --model-id     Base model ID. Default: Qwen/Qwen3-8B"
    echo "  -d, --dataset      Path to dataset. Default: data/passport_grpo.jsonl"
    echo "  -o, --output-dir   Output directory for model. Default: Qwen3-8B-GRPO"
    echo "  -l, --lr           Learning rate. Default: 1e-5"
    echo "  -e, --epochs       Number of training epochs. Default: 2"
    echo "  -b, --batch-size   Batch size/gradient accumulation steps. Default: 16"
    echo "  -m, --method       Training method. Default: transformers"
    echo "  -h, --help         Show this help message"
}

# Parse command line arguments
ARGS=$(getopt -o i:d:o:l:e:b:m:h --long model-id:,dataset:,output-dir:,lr:,epochs:,batch-size:,method:,help -n "$0" -- "$@")

# Check for invalid arguments
if [ $? -ne 0 ]; then
    usage
    exit 1
fi

# Process arguments
eval set -- "$ARGS"
while true; do
    case "$1" in
        -i|--model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            # Validate method
            if [ "$METHOD" != "transformers" ]; then
                echo "Error: Method must be 'transformers'"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

# Run the Python script with the specified arguments
python src/grpo_stage.py \
    --model-id "$MODEL_ID" \
    --dataset "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --method "$METHOD"