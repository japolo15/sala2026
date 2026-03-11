import cv2
import os
import numpy as np

def run_image_pipeline():
    # 1. Configuration
    input_video = os.path.join('hackathon_data', 'bruv-videos', 'LGH020002.MP4')
    mask_path = 'static_bruv_mask.png'
    output_folder = 'processed_frames'
    frame_interval = 4
    target_w, target_h = 1280, 720

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # 2. Check for inputs
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found at {mask_path}. Please ensure it exists in the main directory.")
        return

    # 3. Initialize Video Capture and Mask
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"--- Starting Image Sequence Pipeline ---")
    print(f"Input: {input_video}")
    print(f"Output Folder: {output_folder}/")
    print(f"Processing every {frame_interval}th frame (Inpaint -> CLAHE -> 720p Downscale)")

    # 4. Process Frames
    frame_idx = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every 4th frame
            if frame_idx % frame_interval == 0:
                # A. Inpaint at ORIGINAL resolution
                inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                
                # B. CLAHE at ORIGINAL resolution
                lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                # C. Downscale to 720p AFTER processing
                processed_frame = cv2.resize(enhanced_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                # D. Save as PNG
                output_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(output_path, processed_frame)
                
                saved_count += 1

                if saved_count % 50 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({saved_count} images saved)", end='\r')

            frame_idx += 1

    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        cap.release()
        print(f"\nPipeline complete!")
        print(f"Total processed and saved: {saved_count} images in '{output_folder}/'")

if __name__ == "__main__":
    run_image_pipeline()
