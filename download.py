import r2_download as hd
import os

os.environ["R2_ENDPOINT"] = "https://6200702e94592ad231a53daba00f8a5d.r2.cloudflarestorage.com"
os.environ["R2_ACCESS_KEY_ID"] = "93bb95ebfe47d5ef93c45efe3c108ca8"
os.environ["R2_SECRET_ACCESS_KEY"] = "cee49fead9c1a8ac2741a4c2703c908efc5d965100a2d8d20c233fce05547a55"
os.environ["R2_BUCKET"] = "sala-2026-hackathon-data"

client = hd.get_s3_client()
manifest = hd.load_manifest(bucket=os.environ["R2_BUCKET"], s3_client=client, cache_path="manifest.json")

# Download just one sub-video to start (~4 GB)
stats = hd.download_dataset(manifest, dataset_name="bruv-videos", tags=["vid2-sub02"])