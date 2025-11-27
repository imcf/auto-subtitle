# auto-subtitle

Lightweight CLI to generate and embed subtitles in video files using WhisperX and ffmpeg.

## Quick Start

Generate SRT and burn subtitles into an MP4:

```fish
python -m auto_subtitle.cli video.mp4 --device cuda --subtitle_mode burn -o output/
```

Use an existing SRT and embed as a track:

```fish
python -m auto_subtitle.cli video.mp4 --srt_path video.srt --subtitle_mode embed -o output/
```

## Installation

### System Requirements

- Python 3.8+
- ffmpeg (for audio extraction and video processing)
- CUDA toolkit (optional, for GPU acceleration)

### Install ffmpeg

```fish
sudo apt update && sudo apt install -y ffmpeg
```

### Install Python Dependencies

For GPU support (recommended), install PyTorch with CUDA first:

```fish
# Example for CUDA 12.8 - adjust for your system
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the package:

```fish
pip install -r requirements.txt
pip install -e .
```

For CPU-only:

```fish
pip install -r requirements.txt
pip install -e .
```

## Usage

Basic command structure:

```fish
python -m auto_subtitle.cli <video_files> [options]
```

### Common Options

- `--model <name>` - WhisperX model (default: `small`)
- `--device <cpu|cuda>` - Processing device (default: auto-detect)
- `--language <code>` - Language code (default: `auto`)
- `--subtitle_mode <burn|embed|external>` - How to add subtitles
  - `burn` - Hardcode into video (default)
  - `embed` - Add as subtitle track
  - `external` - Copy video without subtitles
- `--output_dir/-o <path>` - Output directory (default: `.`)
- `--output_srt` - Save SRT files next to videos
- `--srt_only` - Generate SRT files only, skip video processing
- `--edit_srt` - Open SRTs in editor before processing
- `--editor <cmd>` - Editor command (e.g., `"code --wait"`)

### Batch Processing

Process multiple files:

```fish
python -m auto_subtitle.cli video1.mp4 video2.mp4 --device cuda -o output/
```

Process directory:

```fish
python -m auto_subtitle.cli --input_dir videos/ --device cuda -o output/
```

Recursive directory scan:

```fish
python -m auto_subtitle.cli --input_dir videos/ --recursive --device cuda -o output/
```

### Examples

Generate SRT files only:

```fish
python -m auto_subtitle.cli video.mp4 --srt_only --output_srt
```

Edit subtitles before burning:

```fish
python -m auto_subtitle.cli video.mp4 --edit_srt --editor "nano" --subtitle_mode burn -o output/
```

Use large model with translation:

```fish
python -m auto_subtitle.cli video.mp4 --model large --task translate --device cuda -o output/

### GPU accelerated subtitle burning

When `--device cuda` is used, `auto-subtitle` will attempt to accelerate the video encoding step
using NVIDIA NVENC (`h264_nvenc`) where available. Note that ffmpeg's `subtitles` filter renders
subtitle overlays using the CPU; however, `h264_nvenc` reduces the cost of video encoding and
significantly speeds up the overall process on supported NVIDIA GPUs.

Usage example (hardware-accelerated encoding with NVENC):

```fish
python -m auto_subtitle.cli video.mp4 --device cuda --subtitle_mode burn -o output/
```

If `h264_nvenc` is not available in your ffmpeg build, `auto-subtitle` will fall back to `libx264`.
If an NVENC-accelerated attempt fails at runtime, a fallback to CPU encoding (`libx264`) will be used
and a warning will be printed.

Note: Your system needs an ffmpeg build with NVENC (h264_nvenc) enabled for CUDA acceleration to work.
Defaults and tuning
-------------------

By default `auto-subtitle` will encode output videos with higher quality defaults than before:

- libx264: CRF=18, preset=medium, pixel format yuv420p (good visual quality, larger file sizes)
- NVENC: preset p4, vbr_hq with CQ=19, pixel format yuv420p

You can tune these values at the CLI with:

```fish
python -m auto_subtitle.cli video.mp4 --device cuda --subtitle_mode burn --crf 20 --nvenc_cq 22 -o output/
```

Lower `--crf` or `--nvenc_cq` values result in higher quality but larger files. Use higher values (e.g., 22-28) to reduce file size.

On many Linux distros, the default `ffmpeg` package may be compiled without NVENC; you can either
install a distribution-provided ffmpeg with NVENC support or compile ffmpeg from source following
NVIDIA's documentation (e.g., enabling `--enable-nvenc` and correct CUDA SDK paths).
```

## Troubleshooting

### Memory Issues

If you encounter `std::bad_alloc` errors:

1. Use CPU mode: `--device cpu`
2. Use a smaller model: `--model tiny` or `--model base`
3. Ensure sufficient RAM (8GB+ recommended for GPU, 4GB+ for CPU)
4. Check that ffmpeg is installed correctly

### GPU Issues

If GPU is not detected:

1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA availability:

   ```fish
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Reinstall PyTorch with correct CUDA version

## License

See `LICENSE` file in this repository.
