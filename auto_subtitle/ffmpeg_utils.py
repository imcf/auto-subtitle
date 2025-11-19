import os
import subprocess
import ffmpeg
import tempfile
from pathlib import Path
from typing import Any, Optional


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


def _run_ffmpeg_cli_and_log(args: list, cwd: Optional[str] = None, verbose: bool = False):
    if verbose:
        print("Running ffmpeg command:", " ".join(args))
    try:
        result = subprocess.run(args, capture_output=not verbose, text=True, check=True, cwd=cwd)
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
    return "'" + value.replace("'", "\'") + "'"


def add_subtitles_to_video(input_path: str, srt_path: str, out_path: str, verbose: bool = False, mode: str = "burn") -> None:
    """Add subtitles to a video file using ffmpeg.

    mode: 'burn' = hardcode, 'embed' embed as track, 'external' do not add subtitles
    """
    if mode == "external":
        try:
            cmd = ffmpeg.input(input_path).output(out_path, vcodec="copy", acodec="copy")
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error:
            cmd = ffmpeg.input(input_path).output(out_path, vcodec="libx264", acodec="aac")
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
            "subtitles", filename=normalized_srt, force_style="OutlineColour=&H40000000,BorderStyle=3"
        )
        cmd = ffmpeg.output(filtered_video, audio, out_path, vcodec="libx264", acodec="aac")
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
                    filter_value = _quote_for_ffmpeg_filter(os.path.basename(fallback_srt))
                    filter_arg = f"subtitles={filter_value}"
                    args = [
                        "ffmpeg", "-i", input_path,
                        "-vf", filter_arg,
                        "-c:v", "libx264", "-c:a", "aac", out_path, "-y"
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
            cmd = ffmpeg.output(video, srt_input, out_path, vcodec="copy", acodec="copy", scodec=scodec)
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error:
            cmd = ffmpeg.output(video, srt_input, out_path, vcodec="libx264", acodec="aac", scodec=scodec)
            _run_ffmpeg_and_log(cmd, verbose=verbose)
