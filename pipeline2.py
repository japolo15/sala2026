import cv2
import subprocess
import os
import sys

def run_optimized_pipeline():
    # 1. Configuration (Adapted to project paths)
    input_video = os.path.join('hackathon_data', 'bruv-videos', 'LGH020002.MP4')
    mask_path = 'static_bruv_mask.png'
    output_video = 'sam_ready_video.mp4'
    frame_interval = 4

    # 2. Check for inputs
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found at {mask_path}. Please create it using pipeline.py or prepare_bruv.py first.")
        return

    # 3. Initialize Video Capture and Mask
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set target dimensions to 720p for output
    target_w, target_h = 1280, 720

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read mask at {mask_path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate new FPS to maintain original duration
    new_fps = orig_fps / frame_interval
    fps_arg = f"{new_fps:.3f}"

    print(f"Input: {input_video} (@ {orig_fps:.2f} FPS)")
    print(f"Processing at original resolution, then downscaling to {target_w}x{target_h} @ {fps_arg} FPS")
    print(f"Syncing timeline by processing every {frame_interval}th frame...")

    # 4. Setup FFmpeg Pipe
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f"{target_w}x{target_h}",
        '-framerate', fps_arg,
        '-i', '-',
        '-c:v', 'mpeg4',
        '-q:v', '1',
        '-pix_fmt', 'yuv420p',
        output_video
    ]

    # Open the pipe (stderr is enabled for console logging)
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # 5. Process Frames
    frame_idx = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process and write every 4th frame
            if frame_idx % frame_interval == 0:
                # 1. Apply Telea Inpainting at ORIGINAL resolution
                inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                
                # 2. Apply CLAHE Enhancement at ORIGINAL resolution
                lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                # Convert back to BGR
                enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                # 3. Downscale to 720p AFTER processing
                processed_frame = cv2.resize(enhanced_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                # Write raw BGR bytes to the ffmpeg pipe
                process.stdin.write(processed_frame.tobytes())
                processed_count += 1

                if processed_count % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({processed_count} frames written)", end='\r')

            frame_idx += 1

    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        # 6. Cleanup
        cap.release()
        if process.stdin:
            process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            print(f"\nSuccessfully created: {output_video}")
            print(f"Total original frames: {frame_idx}")
            print(f"Total processed frames: {processed_count}")
        else:
            print(f"\nFFmpeg process exited with error code {process.returncode}")

if __name__ == "__main__":
    run_optimized_pipeline()
