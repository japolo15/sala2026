import subprocess
import os
from pathlib import Path

# Configuration
video_path = 'C:\\Users\\japol\\Documents\\codingStuff\\hackathon\\sala2026\\hackathon_data\\bruv-videos\\LGH020002.MP4'
output_folder = 'extracted_frames'
frame_interval = 4  # Extract every 4th frame

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Check if ffmpeg is available
try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print("FFmpeg found!")
    print(f"Version: {result.stdout.split()[2]}")
except FileNotFoundError:
    print("Error: FFmpeg not found. Please install FFmpeg:")
    print("- Windows: Download from https://ffmpeg.org/download.html")
    print("- Or use: winget install ffmpeg")
    exit(1)

# Get video info first
print("\nGetting video information...")
probe_cmd = [
    'ffprobe',
    '-v', 'error',
    '-count_frames',
    '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames,r_frame_rate',
    '-of', 'default=noprint_wrappers=1',
    video_path
]

try:
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error getting video info: {e}")

# FFmpeg command to extract every Nth frame
# -i: input file
# -vf "select='not(mod(n\,4))'" : select every 4th frame (n starts at 0)
# -vsync vfr : variable frame rate (only output selected frames)
# -q:v 1 : quality for PNG (1 is best, ignored for PNG but kept for compatibility)
print(f"\nExtracting every {frame_interval}th frame...")
print(f"Output: {output_folder}/frame_%06d.png\n")

ffmpeg_cmd = [
    'ffmpeg',
    '-i', video_path,
    '-vf', f"select='not(mod(n\\,{frame_interval}))'",
    '-vsync', 'vfr',
    '-frame_pts', '1',  # Use original frame numbers
    os.path.join(output_folder, 'frame_%06d.png')
]

# Run FFmpeg
try:
    subprocess.run(ffmpeg_cmd, check=True)
    print("\n✓ Extraction complete!")
    
    # Count extracted frames
    frame_count = len(list(Path(output_folder).glob('*.png')))
    print(f"✓ Extracted {frame_count} frames to '{output_folder}/' folder.")
    
except subprocess.CalledProcessError as e:
    print(f"\n✗ Error during extraction: {e}")
    exit(1)
