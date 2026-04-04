"""Download all required models to ./models/ for offline use."""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS = {
    "models/mt0-large": "bigscience/mt0-large",
    "models/bartpho-syllable": "vinai/bartpho-syllable",
}


def main():
    for local_path, hub_id in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Downloading {hub_id} → {local_path}")
        print("=" * 50)
        AutoTokenizer.from_pretrained(hub_id).save_pretrained(local_path)
        AutoModelForSeq2SeqLM.from_pretrained(hub_id).save_pretrained(local_path)
        print(f"Done: {local_path}")

    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()
