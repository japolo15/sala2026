import os

import r2_download as hd


DATASET_NAME = "bruv-videos"
ENV_FILE = "participant-download.env"
REQUIRED_ENV_VARS = [
	"R2_ENDPOINT",
	"R2_ACCESS_KEY_ID",
	"R2_SECRET_ACCESS_KEY",
	"R2_BUCKET",
]


def load_env_file(path=ENV_FILE):
	"""Load KEY=VALUE pairs from a .env-style file into process environment."""
	if not os.path.exists(path):
		return

	with open(path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue

			if line.startswith("export "):
				line = line[len("export "):]

			if "=" not in line:
				continue

			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key and value:
				os.environ.setdefault(key, value)


def validate_required_env_vars(required_vars=REQUIRED_ENV_VARS):
	"""Raise a clear error when required R2 credentials are missing."""
	missing = [name for name in required_vars if not os.environ.get(name)]
	if not missing:
		return

	missing_text = ", ".join(missing)
	raise RuntimeError(
		"Missing required environment variable(s): "
		f"{missing_text}. "
		"Create participant-download.env from participant-download.env.example "
		"and fill in your credentials."
	)


def get_subvideo_tags(manifest, dataset_name):
	"""Return sorted unique subvideo tags like vid1-sub02 for a dataset."""
	datasets = manifest.get("datasets", {})
	dataset = datasets.get(dataset_name, {})
	shards = dataset.get("shards", [])

	tags = set()
	for shard in shards:
		for tag in shard.get("tags", []):
			if tag.startswith("vid") and "-sub" in tag:
				tags.add(tag)

	return sorted(tags)


def prompt_choice(valid_tags):
	"""Prompt for download mode and return selected tags or None for full dataset."""
	print("Choose what to download:")
	print("1) One BRUV subvideo")
	print("2) Entire BRUV dataset")

	while True:
		choice = input("Enter 1 or 2: ").strip()

		if choice == "2":
			return None

		if choice == "1":
			print("\nAvailable subvideo tags:")
			print(", ".join(valid_tags))
			while True:
				tag = input("Enter subvideo tag (example: vid2-sub02): ").strip()
				if tag in valid_tags:
					return [tag]
				print("Invalid tag. Please enter one of the listed tags.")

		print("Invalid choice. Please type 1 or 2.")


def main():
	load_env_file()
	validate_required_env_vars()

	client = hd.get_s3_client()
	manifest = hd.load_manifest(
		bucket=os.environ["R2_BUCKET"],
		s3_client=client,
		cache_path="manifest.json",
	)

	valid_tags = get_subvideo_tags(manifest, DATASET_NAME)
	if not valid_tags:
		raise RuntimeError(f"No subvideo tags found for dataset '{DATASET_NAME}'.")

	tags = prompt_choice(valid_tags)
	stats = hd.download_dataset(
		manifest,
		dataset_name=DATASET_NAME,
		tags=tags,
	)

	print("\nDownload complete.")
	print(
		f"Downloaded: {stats['downloaded']}, "
		f"Skipped: {stats['skipped']}, "
		f"Failed: {stats['failed']}"
	)
	if stats["errors"]:
		print("Errors:")
		for item in stats["errors"]:
			print(f"- {item['key']}: {item['error']}")


if __name__ == "__main__":
	main()