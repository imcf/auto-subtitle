import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import ffmpeg


def _ffmpeg_supports_subtitles() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, check=True
        )
        return "subtitles" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run_ffmpeg_and_log(cmd: Any, verbose: bool = False):
    try:
        cmd.run(capture_stdout=not verbose, capture_stderr=True, overwrite_output=True)
    except ffmpeg.Error as e:
        stdout = e.stdout.decode(errors="ignore") if e.stdout else ""
        stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
        print("ffmpeg failed. stdout:")
        if stdout:
            print(stdout)
        print("ffmpeg failed. stderr:")
        if stderr:
            print(stderr)
        raise


def _run_ffmpeg_cli_and_log(
    args: list, cwd: Optional[str] = None, verbose: bool = False
):
    if verbose:
        print("Running ffmpeg command:", " ".join(args))
    try:
        result = subprocess.run(
            args, capture_output=not verbose, text=True, check=True, cwd=cwd
        )
        if verbose and result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print("ffmpeg failed. stdout:")
            print(e.stdout)
        if e.stderr:
            print("ffmpeg failed. stderr:")
            print(e.stderr)
        raise


def _quote_for_ffmpeg_filter(value: str) -> str:
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    return "'" + value.replace("'", "'") + "'"


def _ffmpeg_supports_encoder(encoder: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        return encoder in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def add_subtitles_to_video(
    input_path: str,
    srt_path: str,
    out_path: str,
    verbose: bool = False,
    mode: str = "burn",
    device: Optional[str] = None,
    crf: int = 18,
    nvenc_cq: int = 19,
    nvenc_preset: str = "p4",
    nvenc_lookahead: int = 32,
    x264_preset: str = "medium",
    audio_bitrate: str = "192k",
) -> None:
    """Add subtitles to a video file using ffmpeg.

    Parameters
    ----------
    input_path : str
        Path to input video file.
    srt_path : str
        Path to SRT file containing subtitles.
    out_path : str
        Path to output video file to write.
    verbose : bool
        Print ffmpeg debug output if True.
    mode : str
        One of 'burn', 'embed', or 'external' to control how subtitles are added.
        - 'burn' : hardcode subtitles into the output video
        - 'embed' : add subtitles as a selectable track
        - 'external' : do not add subtitles; just copy or re-encode the input
    device : Optional[str]
        If 'cuda', attempt to use ffmpeg's NVENC encoder (h264_nvenc) and CUDA
        hardware acceleration for the encoding step. Subtitle rendering still
        occurs on the CPU via the 'subtitles' filter; using NVENC reduces video
        encoding cost and speeds up the overall process on NVIDIA GPUs.
    """
    if mode == "external":
        try:
            cmd = ffmpeg.input(input_path).output(
                out_path, vcodec="copy", acodec="copy"
            )
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error:
            cmd = ffmpeg.input(input_path).output(
                out_path,
                vcodec="libx264",
                acodec="aac",
                crf=crf,
                preset="medium",
                pix_fmt="yuv420p",
                movflags="faststart",
            )
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        return

    if not _ffmpeg_supports_subtitles() and mode == "burn":
        print("Warning: this ffmpeg build may not support the 'subtitles' filter.")

    input_path = str(Path(input_path).resolve())
    out_path = str(Path(out_path).resolve())
    srt_path = str(Path(srt_path).resolve())
    normalized_srt = Path(srt_path).as_posix()

    video = ffmpeg.input(input_path)
    audio = video.audio

    if mode == "burn":
        filtered_video = video.filter(
            "subtitles",
            filename=normalized_srt,
            force_style="OutlineColour=&H40000000,BorderStyle=3",
        )

        # If CUDA is available and ffmpeg provides h264_nvenc, use NVENC for encoding
        # to accelerate the process. Note: the subtitles filter is CPU-bound in ffmpeg,
        # but NVENC as the encoder and CUDA hwaccel / hwupload can reduce encode cost.
        use_nvenc = False
        if (
            device
            and device.lower() == "cuda"
            and _ffmpeg_supports_encoder("h264_nvenc")
        ):
            use_nvenc = True

        if use_nvenc:
            if verbose:
                print(
                    "Using NVENC encoding (h264_nvenc) with CUDA hwaccel for subtitle burn"
                )
            # Prefer constructing an ffmpeg CLI that enables cuda hwaccel and NVENC
            # to accelerate encoding. We use the subtitles filter as usual and pass
            # the resulting frames to the GPU-backed encoder.
            filter_value = _quote_for_ffmpeg_filter(normalized_srt)
            # Format -> hwupload_cuda ensures frames are uploaded to the GPU for NVENC
            vf_chain = f"subtitles={filter_value},format=nv12,hwupload_cuda"
            args = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-i",
                input_path,
                "-vf",
                vf_chain,
                "-c:v",
                "h264_nvenc",
                # Quality settings for NVENC: use vbr_hq with a low CQ value for
                # better perceived quality. Pix fmt and faststart for compatibility.
                "-preset",
                nvenc_preset,
                "-rc:v",
                "vbr_hq",
                "-cq",
                str(nvenc_cq),
                "-rc-lookahead",
                str(nvenc_lookahead),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                out_path,
            ]
            try:
                _run_ffmpeg_cli_and_log(args, verbose=verbose)
                return
            except Exception:
                # If NVENC path fails, fall back to the ffmpeg-python API with libx264
                print(
                    "Warning: NVENC accelerated encoding failed, falling back to CPU libx264"
                )

        cmd = ffmpeg.output(
            filtered_video,
            audio,
            out_path,
            vcodec="libx264",
            acodec="aac",
            audio_bitrate=audio_bitrate,
            # crf=18 is visually lossless or near-lossless; preset=medium balances
            # speed vs compression. pix_fmt yuv420p for compatibility and faststart
            # to enable streaming-friendly MP4 files.
            crf=crf,
            preset=x264_preset,
            pix_fmt="yuv420p",
            movflags="faststart",
        )
        try:
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error as e:
            stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
            if "Unable to open" in stderr and ":\\" in srt_path:
                fallback_dir = tempfile.mkdtemp()
                fallback_srt = os.path.join(fallback_dir, os.path.basename(srt_path))
                try:
                    with open(srt_path, "rb") as src, open(fallback_srt, "wb") as dst:
                        dst.write(src.read())
                    filter_value = _quote_for_ffmpeg_filter(
                        os.path.basename(fallback_srt)
                    )
                    filter_arg = f"subtitles={filter_value}"
                    args = [
                        "ffmpeg",
                        "-i",
                        input_path,
                        "-vf",
                        filter_arg,
                        "-c:v",
                        "libx264",
                        "-crf",
                        str(crf),
                        "-preset",
                        x264_preset,
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        out_path,
                        "-y",
                    ]
                    _run_ffmpeg_cli_and_log(args, cwd=fallback_dir, verbose=verbose)
                finally:
                    try:
                        os.remove(fallback_srt)
                    except Exception:
                        pass
                    try:
                        os.rmdir(fallback_dir)
                    except Exception:
                        pass
            raise

    elif mode == "embed":
        ext = Path(out_path).suffix.lower()
        if ext in [".mp4", ".mov", ".m4v"]:
            scodec = "mov_text"
        else:
            scodec = "subrip"
        srt_input = ffmpeg.input(srt_path)
        try:
            cmd = ffmpeg.output(
                video, srt_input, out_path, vcodec="copy", acodec="copy", scodec=scodec
            )
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error:
            cmd = ffmpeg.output(
                video,
                srt_input,
                out_path,
                vcodec="libx264",
                acodec="aac",
                scodec=scodec,
                crf=crf,
                preset="medium",
                pix_fmt="yuv420p",
                movflags="faststart",
            )
            _run_ffmpeg_and_log(cmd, verbose=verbose)
