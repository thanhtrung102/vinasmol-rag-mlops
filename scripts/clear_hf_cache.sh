#!/bin/bash
# Clear HuggingFace transformers cache to resolve model loading issues
#
# Usage: ./scripts/clear_hf_cache.sh [model_name]
#
# Examples:
#   ./scripts/clear_hf_cache.sh                    # Clear all PhoGPT models
#   ./scripts/clear_hf_cache.sh vinai/PhoGPT-4B-Chat  # Clear specific model

set -e

MODEL_NAME="${1:-vinai/PhoGPT}"

echo "Clearing HuggingFace cache for models matching: $MODEL_NAME"

# Clear transformers modules cache
CACHE_DIR="$HOME/.cache/huggingface/modules/transformers_modules"
if [ -d "$CACHE_DIR" ]; then
    echo "Checking transformers modules cache: $CACHE_DIR"

    # Convert model name to directory pattern (e.g., vinai/PhoGPT -> PhoGPT*)
    MODEL_DIR_PATTERN=$(echo "$MODEL_NAME" | sed 's/.*\///')

    find "$CACHE_DIR" -type d -name "${MODEL_DIR_PATTERN}*" -print0 | while IFS= read -r -d '' dir; do
        echo "Removing: $dir"
        rm -rf "$dir"
    done
else
    echo "Transformers modules cache not found: $CACHE_DIR"
fi

# Clear hub cache
HUB_CACHE_DIR="$HOME/.cache/huggingface/hub"
if [ -d "$HUB_CACHE_DIR" ]; then
    echo "Checking hub cache: $HUB_CACHE_DIR"

    # Find model directories in hub cache
    find "$HUB_CACHE_DIR" -type d -name "*${MODEL_DIR_PATTERN}*" -print0 | while IFS= read -r -d '' dir; do
        echo "Removing: $dir"
        rm -rf "$dir"
    done
else
    echo "Hub cache not found: $HUB_CACHE_DIR"
fi

echo "Cache cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Run your training script again"
echo "  2. The model will be re-downloaded with fresh files"
