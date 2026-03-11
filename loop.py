import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

import cv2 as cv

# Configuration
# Use a generic path that works across platforms (Windows/Linux)
video_path = os.path.join('hackathon_data', 'bruv-videos', 'LGH020002.MP4')
output_folder = 'extracted_frames'
frame_interval = 4  # Extract every 4th frame

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# 1. Create a VideoCapture object
cap = cv.VideoCapture(video_path)

# 2. Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video or camera.")
    exit()

# Get video properties
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# print(f"Video FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Extracting every {frame_interval}th frame...")

# 3. Read and save frames
frame_count = 0
saved_count = 0

while True:
    # 'ret' is a boolean, True if frame was read correctly
    # 'frame' is the actual image frame (a NumPy array)
    ret, frame = cap.read()

    # Break the loop if the frame was not read correctly (e.g., end of video)
    if not ret:
        print(f"Extraction complete. Saved {saved_count} frames.")
        break
    
    print(f"Processing frame {frame_count}")

    # Save every 4th frame
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:06d}.png')
        # cv.imwrite(frame_filename, frame)
        saved_count += 1
        
        # Print progress every 100 saved frames
        if saved_count % 100 == 0:
            print(f"Saved {saved_count} frames...")
    
    frame_count += 1

# 4. Release the video capture object
cap.release()
print(f"Done! Extracted {saved_count} frames to '{output_folder}/' folder.")
