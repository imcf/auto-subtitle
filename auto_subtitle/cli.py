import argparse
                    try:
                        result = model.transcribe(audio_path, **transcribe_kwargs)
                    except Exception as e:
                        msg = str(e)
                        print(f"Error: failed to transcribe {audio_path} with the selected backend: {e}")
                        # If this is a CUDA runtime issue (missing DLL or CUDA
                        # library), try falling back to CPU to continue.
                        if ("cublas" in msg.lower() or "cuda" in msg.lower()) and device != "cpu":
                            print("Falling back to CPU because the CUDA runtime is unavailable or misconfigured.")
                            try:
                                # Reload the model on CPU and retry
                                if backend == "whisperx":
                                    cpu_model = whisperx.load_model(model_name, device="cpu")
                                    result = cpu_model.transcribe(audio_path, **transcribe_kwargs)
                                else:
                                    cpu_model = openai_whisper.load_model(model_name, device="cpu")
                                    result = cpu_model.transcribe(audio_path, **transcribe_kwargs)
                                # If the backend supports alignment, align on CPU too
                                if backend == "whisperx":
                                    align_model, metadata = whisperx.load_align_model(
                                        language_code=result.get("language"), device="cpu"
                                    )
                                    aligned_segments = whisperx.align(
                                        result["segments"], align_model, metadata, audio, "cpu"
                                    )
                                    return {"segments": aligned_segments, "language": result.get("language")}
                                return result
                            except Exception as e2:
                                print(f"Error: failed to transcribe on CPU as fallback: {e2}")
                                raise
                        raise
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
    model_choices = openai_whisper.available_models() if openai_whisper else [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large",
        "large-v2",
    ]
    parser.add_argument(
        "--model",
        default="small",
        choices=model_choices,
        help="name of the Whisper model to use (models available from OpenAI Whisper)",
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
        help="whether to output the .srt file next to the original video file(s) (same basename)",
    )
    parser.add_argument(
        "--srt_only",
        type=str2bool,
        default=False,
        help="only generate the .srt file and not create overlayed video",
    )
    parser.add_argument(
        "--subtitle_mode",
        type=str,
        default="burn",
        choices=["burn", "embed", "external"],
        help="What to do with the generated subtitles: burn (hardcode), embed (add as track), or external (create .srt only)",
    )
    parser.add_argument(
        "--max_chars_per_line",
        type=int,
        default=42,
        help="Maximum number of characters per subtitle line (word-wrapped).",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=2,
        help="Maximum number of lines per subtitle entry.",
    )
    parser.add_argument(
        "--max_sub_duration",
        type=float,
        default=5.0,
        help="Maximum duration (seconds) per subtitle entry; longer segments will be split.",
    )
    parser.add_argument(
        "--min_sub_duration",
        type=float,
        default=0.5,
        help="Minimum duration (seconds) per subtitle entry; too-short segments are merged.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="whether to print out the progress and debug messages",
    )
    parser.add_argument(
        "--edit_srt",
        type=str2bool,
        default=False,
        help="Open generated SRT files for manual editing before burning/embedding.",
    )
    parser.add_argument(
        "--editor",
        type=str,
        default=None,
        help="Editor command to use for editing SRT files. Can include args, e.g. 'code --wait'.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="After generating SRTs, run burn on all files in batch mode (non-interactive)",
    )
    # Use --edit_srt to open SRT files in an external editor for manual edits.

    parser.add_argument(
        "--srt_path",
        type=str,
        default=None,
        help=(
            "Path to an existing .srt file (or a directory of .srt files) to use "
            "instead of generating a new one. When a directory is provided, the CLI "
            "looks for files named <basename>.srt for each input video."
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="whisperx",
        choices=["whisperx", "openai-whisper"],
        help="Which transcription backend to use: whisperx (alignment & faster batched inference) or openai-whisper (original whisper).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Force the device used for model inference: 'cpu' or 'cuda'. If unset, the CLI auto-detects CUDA availability.",
    )
    parser.add_argument(
        "--hf_disable_symlinks",
        type=str2bool,
        default=None,
        help=(
            "If True, instruct HuggingFace Hub to disable use of symlinks for cache download "
            "(equivalent to setting HF_HUB_DISABLE_SYMLINKS=1). If not set, the CLI will "
            "retry automatically on Windows when symlink creation fails."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=[
            "auto",
            "af",
            "am",
            "ar",
            "as",
            "az",
            "ba",
            "be",
            "bg",
            "bn",
            "bo",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "gl",
            "gu",
            "ha",
            "haw",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "jw",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "la",
            "lb",
            "ln",
            "lo",
            "lt",
            "lv",
            "mg",
            "mi",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "pa",
            "pl",
            "ps",
            "pt",
            "ro",
            "ru",
            "sa",
            "sd",
            "si",
            "sk",
            "sl",
            "sn",
            "so",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "tt",
            "uk",
            "ur",
            "uz",
            "vi",
            "yi",
            "yo",
            "zh",
        ],
        help="What is the origin language of the video? If unset, it is detected automatically.",
    )

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    verbose: bool = args.pop("verbose")
    subtitle_mode: str = args.pop("subtitle_mode")
    edit_srt: bool = args.pop("edit_srt")
    editor_cmd: str | None = args.pop("editor")
    backend: str = args.pop("backend")
    hf_disable_symlinks: Optional[bool] = args.pop("hf_disable_symlinks")
    # Legacy flags removed from the CLI.
    batch_mode: bool = args.pop("batch")
    max_chars = args.pop("max_chars_per_line")
    max_lines = args.pop("max_lines")
    max_sub_duration = args.pop("max_sub_duration")
    min_sub_duration = args.pop("min_sub_duration")
    srt_path_arg: Optional[str] = args.pop("srt_path")

    # Continue CLI workflows

    # Validate that the provided --editor (if any) points to an executable on PATH.
    if editor_cmd and not _validate_editor_cmd(editor_cmd):
        first = shlex.split(editor_cmd)[0]
        print(
            f"Error: editor executable '{first}' not found in PATH."
            " Please install it or use a different editor command, or omit --editor."
        )
        sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection."
        )
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language

    # Only load a model when we need to transcribe audio (i.e. when
    # --srt_path is not provided); this avoids unnecessary model download when
    # the user only wants to burn/embed existing .srt files.
    model = None
    # The `--device` flag (if provided) overrides auto-detection
    user_device: Optional[str] = args.pop("device")
    if user_device:
        device = user_device
    else:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    # Configure HF hub symlink behavior. Either the user explicitly asked for
    # copying (hf_disable_symlinks True), or we'll let HF hub try symlinks and
    # only retry on error (default behavior).
    if hf_disable_symlinks is None and sys.platform == "win32":
        # On Windows, disable symlinks by default to avoid permission issues in
        # environments where symlink creation is restricted.
        hf_disable_symlinks = True

    if hf_disable_symlinks:
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    if not srt_path_arg:
        if backend == "whisperx":
            if not whisperx:
                raise ImportError("whisperx is not installed. Please install whisperx (and torch) or use --backend openai-whisper.")
            try:
                model = whisperx.load_model(model_name, device=device)
            except OSError as e:
                # Handle Windows permission errors while creating symlinks in the
                # HF cache. When Windows does not allow symlinks for the current
                # user, HuggingFace Hub may attempt to create symlinks and fail
                # with WinError 1314. We can retry after forcing the hub to
                # copy files instead of symlinking.
                if sys.platform == "win32" and (
                    "WinError 1314" in str(e) or "symlink" in str(e).lower()
                ):
                    if verbose:
                        print(
                            "Warning: symlink creation failed while downloading model. "
                            "Retrying with HF_HUB_DISABLE_SYMLINKS=1 (copy files instead of symlink)."
                        )
                    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
                    model = whisperx.load_model(model_name, device=device)
                else:
                    raise
            if verbose:
                print(f"Using backend 'whisperx': loaded model {model_name} on {device}")
        else:
            # Use the original OpenAI whisper backend
            if not openai_whisper:
                raise ImportError("openai-whisper (whisper) is not installed. Please install openai-whisper or use --backend whisperx.")
            model = openai_whisper.load_model(model_name, device=device)
            if verbose:
                print(f"Using backend 'openai-whisper': loaded model {model_name} on {device}")
    videos = args.pop("video")
    input_dir = args.pop("input_dir")
    recursive = args.pop("recursive")
    if input_dir:
        # scan for supported extensions
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

    # If the user gave an existing SRT path (single file or a directory with
    # SRTs), map videos to those files and skip the transcription step.
    subtitles: Dict[str, str]
    if srt_path_arg:
        from pathlib import Path
        provided_path = Path(srt_path_arg)
        subtitles = {}
        if provided_path.is_file():
            # A single SRT file can only be applied to one video input.
            if len(videos) > 1:
                raise ValueError(
                    "When providing a single --srt_path file, only one input video is allowed."
                )
            if not provided_path.exists():
                raise ValueError(f"SRT file not found: {srt_path_arg}")
            subtitles[videos[0]] = str(provided_path.resolve())
        elif provided_path.is_dir():
            for path in videos:
                candidate = provided_path / f"{filename(path)}.srt"
                if not candidate.exists():
                    raise ValueError(
                        f"Missing SRT for video {path}; expected {candidate}"
                    )
                subtitles[path] = str(candidate.resolve())
        else:
            raise ValueError(f"Provided --srt_path does not exist: {srt_path_arg}")
    else:
        audios = get_audio(videos)
        if model is not None:
            task = args.get("task") if "task" in args else "transcribe"
            transcribe_language = language if language != "auto" else None

            if backend == "whisperx":
                def _transcribe_and_align(audio_path: str):
                    transcribe_kwargs = {}
                    if task:
                        transcribe_kwargs["task"] = task
                    if transcribe_language:
                        transcribe_kwargs["language"] = transcribe_language
                    # WhisperX transcribe (batched) followed by alignment
                    audio = whisperx.load_audio(audio_path)
                    try:
                        result = model.transcribe(audio, **transcribe_kwargs)
                    except Exception as e:
                        msg = str(e)
                        print(f"Error: failed to transcribe {audio_path} with whisperX: {e}")
                        # If this is a CUDA runtime issue (missing DLL or cuda
                        # library), try falling back to CPU to continue.
                        if ("cublas" in msg.lower() or "cuda" in msg.lower()) and device != "cpu":
                            print("Falling back to CPU because the CUDA runtime is unavailable or misconfigured.")
                            try:
                                cpu_model = whisperx.load_model(model_name, device="cpu")
                                result = cpu_model.transcribe(audio, **transcribe_kwargs)
                                # Align with CPU model as well
                                align_model, metadata = whisperx.load_align_model(
                                    language_code=result.get("language"), device="cpu"
                                )
                                aligned_segments = whisperx.align(
                                    result["segments"], align_model, metadata, audio, "cpu"
                                )
                                return {"segments": aligned_segments, "language": result.get("language")}
                            except Exception as e2:
                                print(f"Error: failed to transcribe on CPU as fallback: {e2}")
                                raise
                        raise

                    if result is None:
                        raise RuntimeError("whisperx returned no transcription result")

                    try:
                        align_model, metadata = whisperx.load_align_model(
                            language_code=result.get("language"), device=device
                        )
                        aligned_segments = whisperx.align(
                            result["segments"], align_model, metadata, audio, device
                        )
                        return {"segments": aligned_segments, "language": result.get("language")}
                    except Exception as e:
                        if verbose:
                            print(f"Warning: alignment failed for {audio_path}: {e}")
                        # Fall back to the non-aligned result
                        return result

                transcribe_fn = _transcribe_and_align
            else:
                # Original openai-whisper
                def _openai_transcribe(audio_path: str):
                    transcribe_kwargs = {}
                    if task:
                        transcribe_kwargs["task"] = task
                    if transcribe_language:
                        transcribe_kwargs["language"] = transcribe_language
                    # Merge any other remaining args (e.g., temperature overrides) if present
                    transcribe_kwargs.update({k: v for k, v in args.items() if k not in ("task", "language")})
                    return model.transcribe(audio_path, **transcribe_kwargs)

                transcribe_fn = _openai_transcribe

            subtitles = get_subtitles(
                audios,
                output_srt or srt_only,
                output_dir,
                lambda audio_path: transcribe_fn(audio_path),
                max_chars,
                max_lines,
                max_sub_duration,
                min_sub_duration,
            )
        else:
            subtitles = get_subtitles(
                audios,
                output_srt or srt_only,
                output_dir,
                lambda audio_path: None,
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
                path, srt_path, out_path, verbose, mode=subtitle_mode
            )
        return

    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        add_subtitles_to_video(path, srt_path, out_path, verbose, mode=subtitle_mode)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(output_path, acodec="pcm_s16le", ac=1, ar="16k").run(
            quiet=True, overwrite_output=True
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
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        # Save SRT files next to the original video when --output_srt is True.
        if output_srt:
            srt_dir = os.path.dirname(path)
        else:
            srt_dir = tempfile.gettempdir()
        srt_path = os.path.join(srt_dir, f"{filename(path)}.srt")

        print(f"Generating subtitles for {filename(path)}... This might take a while.")

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        try:
            srt_file = open(srt_path, "w", encoding="utf-8")
        except Exception as e:
            # Could not write next to original video (permissions, read-only FS, etc.).
            # Fallback: try writing SRT to output_dir (if set) and then tempdir.
            fallback_dir = output_dir or tempfile.gettempdir()
            fallback_srt = os.path.join(fallback_dir, f"{filename(path)}.srt")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                srt_file = open(fallback_srt, "w", encoding="utf-8")
                print(
                    f"Warning: couldn't write SRT to {srt_path} ({e}); saved to {fallback_srt} instead."
                )
                srt_path = fallback_srt
            except Exception:
                # Final fallback to temp dir
                fallback_dir = tempfile.gettempdir()
                fallback_srt = os.path.join(fallback_dir, f"{filename(path)}.srt")
                srt_file = open(fallback_srt, "w", encoding="utf-8")
                print(
                    f"Warning: couldn't write SRT to {srt_path} or {fallback_dir}; saved to {fallback_srt} instead."
                )

        with srt_file as srt:
            from .utils import write_srt as _write_srt

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
        Path to the file to open
    editor : Optional[str]
        Command to invoke editor. If None, falls back to platform defaults and
        environment variables (EDITOR/VISUAL). If not available, falls back to
        non-blocking open and requests user confirmation.
    verbose : bool
        Whether to print debug information.
    """
    if editor:
        args = shlex.split(editor) + [path]
        if verbose:
            print("Running editor command:", args)
        # Validate the first token exists on PATH to provide a helpful error
        if not shutil.which(args[0]):
            print(f"Warning: editor executable '{args[0]}' not found. Falling back to system defaults.")
            # Use platform defaults
            _open_file_in_editor(path, None, verbose)
            return
        try:
            subprocess.run(args, check=True)
        except FileNotFoundError:
            print(f"Editor '{args[0]}' not found. Falling back to system defaults.")
            # Try platform defaults
            _open_file_in_editor(path, None, verbose)
            return
        # Validate file after editing
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            # Ask user to confirm and possibly re-open
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, editor, verbose)
        return

    env_editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if env_editor:
        args = shlex.split(env_editor) + [path]
        if verbose:
            print("Running editor from environment:", args)
        if not shutil.which(args[0]):
            print(f"Warning: editor executable '{args[0]}' not found in the environment. Falling back to system defaults.")
            _open_file_in_editor(path, None, verbose)
            return
        try:
            subprocess.run(args, check=True)
        except FileNotFoundError:
            print(f"Editor '{args[0]}' not found. Falling back to system defaults.")
            _open_file_in_editor(path, None, verbose)
            return
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, env_editor, verbose)
        return

    # Platform-specific defaults
    if sys.platform == "win32":
        # Notepad blocks until closed
        try:
            subprocess.run(["notepad", path], check=True)
        except FileNotFoundError:
            print("Error: Notepad not found; please set --editor or EDITOR env var to an editor command.")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, None, verbose)
        return
    elif sys.platform == "darwin":
        # open -W waits for the app to close
        try:
            subprocess.run(["open", "-W", path], check=True)
        except FileNotFoundError:
            print("Error: 'open' not found; please set --editor or EDITOR env var to an editor command.")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, None, verbose)
        return
    else:
        # On Linux, prefer a terminal editor if available
        for candidate in ("nano", "vi", "vim"):
            if shutil.which(candidate):
                subprocess.run([candidate, path], check=True)
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    print(f"Warning: {path} is empty or missing after editing.")
                    if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                        _open_file_in_editor(path, None, verbose)
                return
                return
        # fallback: try xdg-open (non-blocking). Ask user to press Enter to continue
        if shutil.which("xdg-open"):
            try:
                subprocess.Popen(["xdg-open", path])
            except Exception:
                print("Failed to launch xdg-open; please set --editor or EDITOR env var.")
            input("Press Enter when finished editing the subtitle file to continue...")
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f"Warning: {path} is empty or missing after editing.")
                if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                    _open_file_in_editor(path, None, verbose)
            return
            return
        # last resort: print a message and return
        print(
            f"Couldn't find an editor; please edit {path} manually and press Enter when done."
        )
        input("Press Enter when finished editing the subtitle file to continue...")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, None, verbose)
        return


def _validate_editor_cmd(editor_cmd: Optional[str]) -> bool:
    """Validate that the given editor command references an existing executable.

    The function checks the first token of the command (e.g. 'code' from
    "code --wait") via `shutil.which`. Returns True if found, False if not.
    """
    if not editor_cmd:
        return True
    try:
        first = shlex.split(editor_cmd)[0]
    except Exception:
        # If shlex fails, be conservative and return False
        return False
    return shutil.which(first) is not None


if __name__ == "__main__":
    main()
