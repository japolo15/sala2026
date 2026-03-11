import subprocess
import os
import cv2
import numpy as np
import glob
from pathlib import Path

def get_video_fps(video_path):
    """Use ffprobe to get the frame rate of the input video."""
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den
        return float(fps_str)
    except Exception as e:
        print(f"Warning: Could not detect FPS, defaulting to 30. Error: {e}")
        return 30.0

def run_pipeline():
    # 1. Configuration
    video_path = os.path.join('hackathon_data', 'bruv-videos', 'LGH020002.MP4')
    extracted_folder = 'extracted_frames'
    inpainted_folder = 'inpainted_frames'
    mask_path = 'static_bruv_mask.png'
    output_video = 'processed_video.mp4'
    frame_interval = 4

    os.makedirs(extracted_folder, exist_ok=True)
    os.makedirs(inpainted_folder, exist_ok=True)

    # 2. Extract Frames using FFmpeg
    print(f"--- Step 1: Extracting every {frame_interval}th frame ---")
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}. Please run download.py first.")
        return

    extract_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"select='not(mod(n\\,{frame_interval}))'",
        '-vsync', 'vfr',
        '-frame_pts', '1',
        os.path.join(extracted_folder, 'frame_%06d.png')
    ]
    
    try:
        subprocess.run(extract_cmd, check=True)
        print("Extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")
        return

    # 3. Inpainting (Logic adapted from inpaint/prepare_bruv.py)
    print("\n--- Step 2: Inpainting ---")
    images = sorted(glob.glob(os.path.join(extracted_folder, "*.png")))
    if not images:
        print("No frames found to process.")
        return

    # Load or create mask
    if os.path.exists(mask_path):
        print(f"Loading existing mask from {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        print("No mask found. Opening interactive masker...")
        first_img = cv2.imread(images[0])
        mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
        drawing = False

        def draw_mask(event, x, y, flags, param):
            nonlocal drawing, mask
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                cv2.circle(mask, (x, y), 15, 255, -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.circle(mask, (x, y), 15, 255, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

        win_name = "Paint over the object to remove. Press 's' to save, 'q' to quit."
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, draw_mask)

        while True:
            overlay = cv2.addWeighted(first_img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imshow(win_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite(mask_path, mask)
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
        cv2.destroyAllWindows()

    print(f"Applying inpainting to {len(images)} frames...")
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        # Using INPAINT_NS (Navier-Stokes) as in the original script
        inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(inpainted_folder, filename), inpainted)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(images)} frames...")

    # 4. Recompose Video using FFmpeg
    print("\n--- Step 3: Composing Video ---")
    original_fps = get_video_fps(video_path)
    # Since we took every 4th frame, we can either keep the original speed (lowering FPS)
    # or keep original FPS (speeding up 4x). We'll keep original FPS for a smoother look.
    
    compose_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(original_fps),
        '-i', os.path.join(inpainted_folder, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        output_video
    ]

    try:
        subprocess.run(compose_cmd, check=True)
        print(f"\nPipeline complete! Output video saved as: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video composition: {e}")

if __name__ == "__main__":
    run_pipeline()
