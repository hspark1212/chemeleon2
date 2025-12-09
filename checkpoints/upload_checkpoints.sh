#!/bin/bash
set -e

# Upload model checkpoints to Hugging Face Hub
# Usage: ./scripts/upload_checkpoints.sh

HF_REPO="hspark1212/chemeleon2-checkpoints"  # TODO: Update with actual HF repo name

echo "🚀 Uploading checkpoints to Hugging Face Hub: $HF_REPO"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ Error: huggingface-cli not found"
    echo "   Install with: pip install huggingface_hub[cli]"
    exit 1
fi

# Check authentication
if [ -z "$HF_TOKEN" ] && ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Error: Not authenticated with Hugging Face"
    echo "   Run: huggingface-cli login"
    echo "   Or set HF_TOKEN environment variable"
    exit 1
fi

echo "📦 Uploading 6 checkpoints..."
echo ""

# Modify with new versioning if needed
# Upload MP-20 VAE
echo "1/6 Uploading MP-20 VAE checkpoint..."
hf upload "$HF_REPO" \
    ckpts/mp_20/vae/dng_m4owq4i5_v0.ckpt \
    v0.0.1/mp_20/vae/dng_m4owq4i5_v0.ckpt

# Upload Alex MP-20 VAE
echo "2/6 Uploading Alex MP-20 VAE checkpoint..."
hf upload "$HF_REPO" \
    ckpts/alex_mp_20/vae/dng_j1jgz9t0_v1.ckpt \
    v0.0.1/alex_mp_20/vae/dng_j1jgz9t0_v1.ckpt

# Upload MP-20 LDM (base)
echo "3/6 Uploading MP-20 LDM (base) checkpoint..."
hf upload "$HF_REPO" \
    ckpts/mp_20/ldm/ldm_null_4tfw67aq.ckpt \
    v0.0.1/mp_20/ldm/ldm_null_4tfw67aq.ckpt

# Upload Alex MP-20 LDM (base)
echo "4/6 Uploading Alex MP-20 LDM (base) checkpoint..."
hf upload "$HF_REPO" \
    ckpts/alex_mp_20/ldm/ldm_null_k744ob4i.ckpt \
    v0.0.1/alex_mp_20/ldm/ldm_null_k744ob4i.ckpt

# Upload MP-20 LDM (RL)
echo "5/6 Uploading MP-20 LDM (RL) checkpoint..."
hf upload "$HF_REPO" \
    ckpts/mp_20/ldm/ldm_rl_dng_xxq3a0mn.ckpt \
    v0.0.1/mp_20/ldm/ldm_rl_dng_xxq3a0mn.ckpt

# Upload Alex MP-20 LDM (RL)
echo "6/6 Uploading Alex MP-20 LDM (RL) checkpoint..."
hf upload "$HF_REPO" \
    ckpts/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt \
    v0.0.1/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt

echo ""
echo "✅ All checkpoints uploaded successfully!"
echo ""
echo "📝 Checksums are already in checkpoints/checkpoints.yaml"
echo "🔗 View at: https://huggingface.co/$HF_REPO"
