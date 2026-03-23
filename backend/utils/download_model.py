from pathlib import Path
from huggingface_hub import snapshot_download


def download_to_local(model_name, output_dir):
    """
    Convert a cached model to a clean local directory structure

    Args:
        model_name: HuggingFace model name (e.g., "Wan-AI/Wan2.1-VACE-1.3B")
        output_dir: Local directory to create clean structure
    """
    try:
        # This will use cache if available, otherwise download
        local_path = snapshot_download(
            repo_id=model_name, local_dir=output_dir, repo_type="model"
        )
        print(f"✅ Successfully converted to: {local_path}")
        return local_path

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    models_to_download = [
        "facebook/sapiens-pretrain-0.3b",
        "facebook/sapiens-pretrain-0.3b-bfloat16",
    ]

    for model_name in models_to_download:
        # Create clean output directory name
        output_dir = Path() / "api/assets/models" / f"{model_name.split('/')[-1]}"
        download_to_local(model_name, output_dir)
        print(f"Files copied to: {output_dir}")
        print()
    else:
        print(f"⚠️  {model_name} not found in cache")


if __name__ == "__main__":
    main()