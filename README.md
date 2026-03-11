# BRUV Video Preprocessing & Segmentation Pipeline

An end-to-end automated system for cleaning, enhancing, and segmenting underwater footage using traditional Computer Vision and Meta's **Segment Anything Model 3 (SAM 3)**.

## 🚀 Achievements & Capabilities
This project delivers a high-performance pipeline specifically tuned for the challenges of marine environments:

*   **Static Object Removal**: Automated inpainting (Telea algorithm) scrubs bait arms and hardware from footage.
*   **Underwater Enhancement**: Integrated **CLAHE** (Contrast Limited Adaptive Histogram Equalization) filters out murky "blue haze," significantly improving object detection rates.
*   **Optimized Memory Management**: 
    *   Replaces slow disk-based image sequences with high-speed **memory-to-ffmpeg piping**.
    *   Intelligent **720p downscaling** ensures state-of-the-art AI models (SAM 3) run on consumer-grade GPUs without "Out of Memory" errors.
*   **Timeline Sync**: Processes every 4th frame while automatically recalculating framerates to keep the visual timeline perfectly synced with the original footage.
*   **Text-Promptable AI**: Leverages SAM 3's latest capabilities to find and segment marine life using natural language prompts (e.g., "fish").

## 🛠 Project Structure

### Core Pipelines
*   `pipeline2.py`: **The Optimized Video-to-Video Pipeline**. Performs inpainting, CLAHE enhancement, and downscaling in memory, outputting a high-quality MP4.
*   `pipeline.py`: **The Image Sequence Pipeline**. Processes frames and exports them as individual PNGs for dataset labeling or manual review.
*   `sam3_image_predictor_example.ipynb`: **AI Inference Notebook**. A Google Colab-ready environment that builds SAM 3, installs dependencies, and runs the integrated preprocessing + segmentation loop.

### Utilities
*   `download.py` / `r2_download.py`: Multi-environment dataset downloader for Cloudflare R2 (supports Colab, Kaggle, RunPod, and Local).
*   `extract_frames_ffmpeg.py`: High-performance frame extraction utility.
*   `manifest.json`: Metadata registry for all hackathon datasets and checksums.

## 💻 Quick Start

### 1. Prepare Data
```bash
python download.py
```

### 2. Run the Optimized Preprocessing
```bash
python pipeline2.py
```

### 3. Run AI Segmentation
1.  Open `sam3_image_predictor_example.ipynb` in Google Colab.
2.  Follow the setup cells to install dependencies.
3.  Run the final cell to process your enhanced frames through the **SAM 3** model.

## 🔧 Technical Specifications
*   **Target Resolution**: 1280x720 (Downscaled from 1080p).
*   **Processing Order**: Inpaint (1080p) -> CLAHE (1080p) -> Resize (720p).
*   **AI Backend**: SAM 3 with Hiera backbone using BFloat16 mixed precision.
*   **Encoding**: High-quality MPEG-4 (fallback for universal compatibility).
