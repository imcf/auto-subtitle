import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from typing import Any, Callable, Dict, Optional

try:
    import whisperx
except ImportError:
    whisperx = None

try:
    from faster_whisper import WhisperModel

    faster_whisper_available = True
except ImportError:
    WhisperModel = None
    faster_whisper_available = False

try:
    import torch
except ImportError:
    torch = None

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

from .ffmpeg_utils import add_subtitles_to_video
from .utils import filename, str2bool


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "video", nargs="*", type=str, help="paths to video files to transcribe"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing video files to process in batch",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan input_dir for video files",
    )
    parser.add_argument(
        "--model",
        default="small",
        type=str,
        help="WhisperX model name (e.g. small, medium)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=".",
        help="directory to save the outputs",
    )
    parser.add_argument(
        "--output_srt",
        type=str2bool,
        default=False,
        help="output .srt files next to videos",
    )
    parser.add_argument(
        "--srt_only", type=str2bool, default=False, help="only generate .srt files"
    )
    parser.add_argument(
        "--subtitle_mode",
        type=str,
        default="burn",
        choices=["burn", "embed", "external"],
        help="burn/embed/external",
    )
    parser.add_argument("--max_chars_per_line", type=int, default=42)
    parser.add_argument("--max_lines", type=int, default=2)
    parser.add_argument("--max_sub_duration", type=float, default=5.0)
    parser.add_argument("--min_sub_duration", type=float, default=0.5)
    parser.add_argument("--verbose", type=str2bool, default=False)
    parser.add_argument("--edit_srt", type=str2bool, default=False)
    parser.add_argument("--editor", type=str, default=None)
    parser.add_argument("--batch", action="store_true", default=False)
    parser.add_argument(
        "--srt_path",
        type=str,
        default=None,
        help="Path to existing .srt file or directory of .srt files",
    )
    parser.add_argument(
        "--task", type=str, default="transcribe", choices=["transcribe", "translate"]
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--hf_disable_symlinks", type=str2bool, default=None)
    parser.add_argument("--language", type=str, default="auto")
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="libx264 CRF (lower values = higher quality, default: 18)",
    )
    parser.add_argument(
        "--nvenc_cq",
        type=int,
        default=19,
        help="NVENC CQ quality level (lower values = higher quality, default: 19)",
    )
    parser.add_argument(
        "--nvenc_preset",
        type=str,
        default="p4",
        help="NVENC preset (p1 best quality -> p7 fastest, default: p4)",
    )
    parser.add_argument(
        "--nvenc_lookahead",
        type=int,
        default=32,
        help="NVENC lookahead to improve ratecontrol (default: 32)",
    )
    parser.add_argument(
        "--x264_preset",
        type=str,
        default="medium",
        help="libx264 preset (veryslow/better compression to ultrafast/fastest), default: medium",
    )
    parser.add_argument(
        "--audio_bitrate",
        type=str,
        default="128k",
        help="Audio bitrate (default: 128k)",
    )

    args = parser.parse_args().__dict__
    model_name = args.pop("model")
    output_dir = args.pop("output_dir")
    output_srt = args.pop("output_srt")
    srt_only = args.pop("srt_only")
    language = args.pop("language")
    verbose = args.pop("verbose")
    subtitle_mode = args.pop("subtitle_mode")
    edit_srt = args.pop("edit_srt")
    editor_cmd = args.pop("editor")
    batch_mode = args.pop("batch")
    max_chars = args.pop("max_chars_per_line")
    max_lines = args.pop("max_lines")
    max_sub_duration = args.pop("max_sub_duration")
    min_sub_duration = args.pop("min_sub_duration")
    srt_path_arg = args.pop("srt_path")
    task = args.pop("task")
    # Determine device early so we can reuse when burning subtitles.
    user_device = args.pop("device")
    device = (
        user_device
        if user_device
        else ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
    )
    crf = args.pop("crf")
    nvenc_cq = args.pop("nvenc_cq")
    nvenc_preset = args.pop("nvenc_preset")
    nvenc_lookahead = args.pop("nvenc_lookahead")
    x264_preset = args.pop("x264_preset")
    audio_bitrate = args.pop("audio_bitrate")

    # Validate editor if provided
    if editor_cmd:
        first = shlex.split(editor_cmd)[0]
        if not shutil.which(first):
            print(f"Error: editor '{first}' not found in PATH.")
            sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection."
        )
        args["language"] = "en"
    elif language != "auto":
        args["language"] = language

    videos = args.pop("video")
    input_dir = args.pop("input_dir")
    recursive = args.pop("recursive")
    if input_dir:
        from pathlib import Path

        exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".m4v"}
        p = Path(input_dir)
        if recursive:
            vids = [str(x) for x in p.rglob("*") if x.suffix.lower() in exts]
        else:
            vids = [str(x) for x in p.iterdir() if x.suffix.lower() in exts]
        videos = vids
    if not videos:
        raise ValueError("No videos provided or found in input_dir")

    # Map provided srt files if given
    subtitles: Dict[str, str] = {}
    if srt_path_arg:
        from pathlib import Path

        provided = Path(srt_path_arg)
        if provided.is_file():
            if len(videos) > 1:
                raise ValueError(
                    "When providing a single --srt_path file, only one input video is allowed."
                )
            subtitles[videos[0]] = str(provided.resolve())
        elif provided.is_dir():
            for path in videos:
                candidate = provided / f"{filename(path)}.srt"
                if not candidate.exists():
                    raise ValueError(
                        f"Missing SRT for video {path}; expected {candidate}"
                    )
                subtitles[path] = str(candidate.resolve())
        else:
            raise ValueError(f"Provided --srt_path does not exist: {srt_path_arg}")
    else:
        audios = get_audio(videos)

        # Load whisperx model (only supported backend)
        if not whisperx and not faster_whisper_available:
            raise ImportError(
                "Neither whisperx nor faster-whisper is installed. "
                "Install with: pip install whisperx OR pip install faster-whisper"
            )

        # Device is already determined above (device variable).

        # Set environment variables to help with memory allocation issues in CTranslate2
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # Determine best compute type based on device
        # Use int8_float16 on GPU, float32 on CPU (int8 types have known issues)
        # See: https://github.com/SYSTRAN/faster-whisper/issues/1260
        if device == "cuda":
            compute_type = "int8_float16"
        else:
            compute_type = "float32"

        # WhisperX has known std::bad_alloc issues that cannot be caught in Python
        # Use faster-whisper directly (more stable, no alignment but works reliably)
        model = None
        use_whisperx = False

        # Skip WhisperX and use faster-whisper directly
        if not faster_whisper_available:
            print("ERROR: faster-whisper is not installed.")
            print("Install with: pip install faster-whisper")
            sys.exit(1)

        if verbose:
            print(
                f"Using faster-whisper with {model_name} model on {device} ({compute_type})"
            )

        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        use_whisperx = False

        if verbose:
            backend = "WhisperX" if use_whisperx else "faster-whisper"
            print(f"Using {backend} with {model_name} model on {device}")

        transcribe_language = None if language == "auto" else language

        def _transcribe_and_align(audio_path: str):
            """Transcribe audio using WhisperX or faster-whisper."""
            if use_whisperx:
                # WhisperX path with alignment
                transcribe_kwargs = {}
                if transcribe_language:
                    transcribe_kwargs["language"] = transcribe_language
                if task:
                    transcribe_kwargs["task"] = task

                # Load and transcribe audio
                audio = whisperx.load_audio(audio_path)
                result = model.transcribe(audio, **transcribe_kwargs)

                if not result or "segments" not in result:
                    raise RuntimeError("Transcription failed or returned no segments")

                # Try alignment for better timing (optional)
                try:
                    detected_lang = result.get("language", transcribe_language or "en")
                    align_model, metadata = whisperx.load_align_model(
                        language_code=detected_lang, device=device
                    )
                    result["segments"] = whisperx.align(
                        result["segments"], align_model, metadata, audio, device
                    )["segments"]
                except Exception as e:
                    if verbose:
                        print(f"Alignment skipped: {e}")

                return result
            else:
                # faster-whisper path (no alignment)
                segments, info = model.transcribe(
                    audio_path,
                    language=transcribe_language,
                    task=task,
                    beam_size=5,
                    vad_filter=True,
                )

                # Convert faster-whisper segments to WhisperX format
                segments_list = []
                for segment in segments:
                    segments_list.append(
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                        }
                    )

                return {
                    "segments": segments_list,
                    "language": info.language
                    if hasattr(info, "language")
                    else transcribe_language,
                }

        subtitles = get_subtitles(
            audios,
            output_srt or srt_only,
            output_dir,
            lambda p: _transcribe_and_align(p),
            max_chars,
            max_lines,
            max_sub_duration,
            min_sub_duration,
        )

    if srt_only:
        # If requested, copy any provided SRTs into the output directory.
        if output_srt:
            for path, srt_path in subtitles.items():
                dst = os.path.join(os.path.dirname(path), f"{filename(path)}.srt")
                try:
                    if os.path.abspath(srt_path) != os.path.abspath(dst):
                        shutil.copyfile(srt_path, dst)
                except Exception as e:
                    print(f"Warning: failed to copy SRT {srt_path} to {dst}: {e}")

        if edit_srt:
            for path, srt_path in subtitles.items():
                print(f"Opening {srt_path} in editor for manual editing...")
                _open_file_in_editor(srt_path, editor_cmd, verbose)
        return

    # If the user wants to manually edit SRT files before burning/embedding,
    # open each SRT in their preferred editor and wait until it closes.
    if edit_srt:
        for path, srt_path in subtitles.items():
            print(f"Opening {srt_path} in editor for manual editing...")
            _open_file_in_editor(srt_path, editor_cmd, verbose)
        # Editing SRTs in the editor is complete.

    # If requested, copy (external) SRTs into the output directory for
    # inspection/share/export when the CLI wasn't responsible for generating
    # them itself.
    if output_srt:
        for path, srt_path in subtitles.items():
            dst = os.path.join(os.path.dirname(path), f"{filename(path)}.srt")
            try:
                if os.path.abspath(srt_path) != os.path.abspath(dst):
                    shutil.copyfile(srt_path, dst)
            except Exception as e:
                print(f"Warning: failed to copy SRT {srt_path} to {dst}: {e}")

    # Continue with CLI-based workflow (edit SRTs with --edit_srt and then
    # optionally burn/embed or export external SRT files).

    if batch_mode:
        # Batch burn all files using the requested mode: subtitle_mode
        for path, srt_path in subtitles.items():
            out_path = os.path.join(output_dir, f"{filename(path)}.mp4")
            print(f"Batch burning {path}...")
            add_subtitles_to_video(
                path,
                srt_path,
                out_path,
                verbose,
                mode=subtitle_mode,
                device=device,
                crf=crf,
                nvenc_cq=nvenc_cq,
                nvenc_preset=nvenc_preset,
                nvenc_lookahead=nvenc_lookahead,
                x264_preset=x264_preset,
                audio_bitrate=audio_bitrate,
            )
        return

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        add_subtitles_to_video(
            path,
            srt_path,
            out_path,
            verbose,
            mode=subtitle_mode,
            device=device,
            crf=crf,
            nvenc_cq=nvenc_cq,
            nvenc_preset=nvenc_preset,
            nvenc_lookahead=nvenc_lookahead,
            x264_preset=x264_preset,
            audio_bitrate=audio_bitrate,
        )

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths):
    """Extract audio from video files to WAV format for transcription.

    Uses memory-efficient flags to avoid std::bad_alloc errors.
    """
    import sys

    temp_dir = tempfile.gettempdir()
    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        sys.stdout.flush()
        base = filename(path)
        output_path = os.path.join(
            temp_dir, f"{base}_{os.getpid()}_{int(time.time())}.wav"
        )

        print("DEBUG: About to run ffmpeg command...", file=sys.stderr)
        sys.stderr.flush()

        # Single-step extraction with memory-efficient flags
        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-analyzeduration",
            "10M",  # INPUT option
            "-probesize",
            "10M",  # INPUT option
            "-i",
            path,
            "-threads",
            "1",  # OUTPUT: single thread
            "-vn",  # OUTPUT: no video
            "-ac",
            "1",  # OUTPUT: mono
            "-ar",
            "16000",  # OUTPUT: 16kHz for Whisper
            "-acodec",
            "pcm_s16le",  # OUTPUT: PCM codec
            "-f",
            "wav",  # OUTPUT: WAV format
            "-map",
            "0:a:0",  # OUTPUT: first audio stream
            output_path,
        ]

        print("DEBUG: Command built, calling subprocess.run...", file=sys.stderr)
        sys.stderr.flush()

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            print(
                f"DEBUG: subprocess.run completed with return code {proc.returncode}",
                file=sys.stderr,
            )
            sys.stderr.flush()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed: {proc.stderr}\n"
                    f"Check that {path} exists and has a valid audio stream."
                )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install with: sudo apt install ffmpeg"
            )

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(
    audio_paths: dict,
    output_srt: bool,
    output_dir: str,
    transcribe: Callable[..., Any],
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    max_sub_duration: float = 5.0,
    min_sub_duration: float = 0.5,
) -> Dict[str, str]:
    """Generate SRT subtitle files from audio transcription.

    Parameters
    ----------
    audio_paths : dict
        Mapping of video paths to extracted audio paths.
    output_srt : bool
        Whether to save SRT next to the original video.
    output_dir : str
        Output directory for subtitles.
    transcribe : Callable
        Function that transcribes audio and returns segments.
    max_chars_per_line : int
        Maximum characters per subtitle line.
    max_lines : int
        Maximum lines per subtitle entry.
    max_sub_duration : float
        Maximum duration for a subtitle segment.
    min_sub_duration : float
        Minimum duration for a subtitle segment.

    Returns
    -------
    Dict[str, str]
        Mapping of video paths to SRT file paths.
    """
    from .utils import write_srt as _write_srt

    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        # Determine output location
        if output_srt:
            srt_path = os.path.join(os.path.dirname(path), f"{filename(path)}.srt")
        else:
            os.makedirs(output_dir, exist_ok=True)
            srt_path = os.path.join(output_dir, f"{filename(path)}.srt")

        print(f"Generating subtitles for {filename(path)}...")

        # Transcribe
        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        # Write SRT
        with open(srt_path, "w", encoding="utf-8") as srt:
            _write_srt(
                result["segments"],
                file=srt,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
                max_duration=max_sub_duration,
                min_duration=min_sub_duration,
            )

        subtitles_path[path] = srt_path

    return subtitles_path


def _open_file_in_editor(
    path: str, editor: Optional[str], verbose: bool = False
) -> None:
    """Open a file in the user's preferred editor and wait for it to close.

    Parameters
    ----------
    path : str
        Path to the file to open.
    editor : Optional[str]
        Editor command. If None, uses EDITOR/VISUAL env vars or platform defaults.
    verbose : bool
        Whether to print debug information.
    """
    # Try user-specified editor
    if editor:
        args = shlex.split(editor) + [path]
        if shutil.which(args[0]):
            if verbose:
                print(f"Running: {' '.join(args)}")
            subprocess.run(args, check=True)
            return

    # Try environment variables
    env_editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if env_editor:
        args = shlex.split(env_editor) + [path]
        if shutil.which(args[0]):
            subprocess.run(args, check=True)
            return

    # Platform-specific defaults
    if sys.platform == "win32":
        subprocess.run(["notepad", path], check=True)
    elif sys.platform == "darwin":
        subprocess.run(["open", "-W", path], check=True)
    else:
        # Linux: try terminal editors first
        for cmd in ["nano", "vi", "vim"]:
            if shutil.which(cmd):
                subprocess.run([cmd, path], check=True)
                return
        # Fallback to xdg-open (non-blocking)
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", path])
            input("Press Enter when finished editing...")


if __name__ == "__main__":
    main()
