import cv2
import numpy as np
import os
import glob

def prepare_bruv():
    # Setup paths
    input_folder = 'input_frames'
    output_folder = 'inpainted_frames'
    mask_path = 'static_bruv_mask.png'

    if not os.path.exists(input_folder):
        print(f"Error: '{input_folder}' directory not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 1. Interactive Masking
    if os.path.exists(mask_path):
        print(f"Existing mask found at '{mask_path}'. Loading it...")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not read existing mask at {mask_path}. Starting interactive masking instead.")
            mask_exists = False
        else:
            mask_exists = True
    else:
        mask_exists = False

    # Find all supported image files
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in extensions:
        # Search case-insensitively if needed, but glob.glob is usually case-sensitive on Linux
        images.extend(glob.glob(os.path.join(input_folder, ext)))
        images.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    # Remove duplicates (in case of case-insensitive matches) and sort
    images = sorted(list(set(images)))

    if not images:
        print(f"No .jpg, .jpeg, or .png images found in '{input_folder}'. Please place some frames there and try again.")
        return

    first_img_path = images[0]
    img = cv2.imread(first_img_path)
    if img is None:
        print(f"Could not read {first_img_path}")
        return

    if not mask_exists:
        # Create a black mask of the same size
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        drawing = False

        # Window name constant
        win_name = "Interactive Masking - Paint over bait arm. Press 's' to save and start inpainting."

        # Create a resizable window
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # Optionally, set an initial reasonable size if the image is huge
        h, w = img.shape[:2]
        screen_w, screen_h = 1280, 720 # Targeted "safe" resolution
        if w > screen_w or h > screen_h:
            scale = min(screen_w / w, screen_h / h)
            cv2.resizeWindow(win_name, int(w * scale), int(h * scale))

        def draw_mask(event, x, y, flags, param):
            nonlocal drawing, mask
            # OpenCV's x,y coordinates in the callback are automatically
            # mapped to the original image resolution even in WINDOW_NORMAL.
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                cv2.circle(mask, (x, y), 15, 255, -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.circle(mask, (x, y), 15, 255, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

        cv2.setMouseCallback(win_name, draw_mask)

        print("\n--- BRUV Frame Preparation ---")
        print(f"Loading first frame: {first_img_path}")
        print("Instructions:")
        print("- CLICK and DRAG to paint the mask (white) over the bait arm.")
        print("- Press 's' to SAVE 'static_bruv_mask.png' and start batch inpainting.")
        print("- Press 'q' to QUIT without saving.")

        while True:
            # Show mask overlayed on image for guidance (0.7 original, 0.3 mask)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # Make the mask red in the overlay for better visibility
            mask_bgr[:,:,0] = 0 # Blue channel
            mask_bgr[:,:,1] = 0 # Green channel

            mask_overlay = cv2.addWeighted(img, 0.8, mask_bgr, 0.5, 0)
            cv2.imshow(win_name, mask_overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite(mask_path, mask)
                print(f"\nMask saved to: {mask_path}")
                break
            elif key == ord('q'):
                print("Quitting...")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    # 2. Batch Inpainting
    print(f"\nStarting batch inpainting on {len(images)} images...")

    for i, img_path in enumerate(images):
        filename = os.path.basename(img_path)
        current_img = cv2.imread(img_path)

        if current_img is None:
            print(f"[{i+1}/{len(images)}] Skipping {filename} (Read Error)")
            continue

        # Apply Inpainting (Navier-Stokes algorithm)
        # inpaintRadius of 3 is usually good for masks
        inpainted = cv2.inpaint(current_img, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, inpainted)

        print(f"[{i+1}/{len(images)}] Processed and saved: {filename}")

    print(f"\nProcessing complete. {len(images)} cleaned frames are in '{output_folder}/'.")

if __name__ == "__main__":
    prepare_bruv()
