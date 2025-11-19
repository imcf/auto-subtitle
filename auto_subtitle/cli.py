import os
import subprocess
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from typing import Dict, Any, Callable
from pathlib import Path
from .utils import filename, str2bool, write_srt


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--subtitle_mode", type=str, default="burn", choices=["burn", "embed", "external"],
                        help="What to do with the generated subtitles: burn (hardcode), embed (add as track), or external (create .srt only)")
    parser.add_argument("--max_chars_per_line", type=int, default=42,
                        help="Maximum number of characters per subtitle line (word-wrapped).")
    parser.add_argument("--max_lines", type=int, default=2,
                        help="Maximum number of lines per subtitle entry.")
    parser.add_argument("--max_sub_duration", type=float, default=5.0,
                        help="Maximum duration (seconds) per subtitle entry; longer segments will be split.")
    parser.add_argument("--min_sub_duration", type=float, default=0.5,
                        help="Minimum duration (seconds) per subtitle entry; too-short segments are merged.")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"],
    help="What is the origin language of the video? If unset, it is detected automatically.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    verbose: bool = args.pop("verbose")
    subtitle_mode: str = args.pop("subtitle_mode")
    max_chars = args.pop("max_chars_per_line")
    max_lines = args.pop("max_lines")
    max_sub_duration = args.pop("max_sub_duration")
    min_sub_duration = args.pop("min_sub_duration")

    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language

    model = whisper.load_model(model_name)
    audios = get_audio(args.pop("video"))
    subtitles = get_subtitles(
        audios, output_srt or srt_only, output_dir, lambda audio_path: model.transcribe(audio_path, **args),
        max_chars, max_lines, max_sub_duration, min_sub_duration
    )

    if srt_only:
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

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths: dict, output_srt: bool, output_dir: str, transcribe: Callable[..., Any],
                  max_chars_per_line: int = 42, max_lines: int = 2, max_sub_duration: float = 5.0, min_sub_duration: float = 0.5) -> Dict[str, str]:
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")

        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(srt_path, "w", encoding="utf-8") as srt:
            from .utils import write_srt as _write_srt
            _write_srt(result["segments"], file=srt, max_chars_per_line=max_chars_per_line,
                       max_lines=max_lines, max_duration=max_sub_duration, min_duration=min_sub_duration)

        subtitles_path[path] = srt_path

    return subtitles_path


def _ffmpeg_supports_subtitles() -> bool:
    """
    Return True if the installed 'ffmpeg' binary supports the `subtitles` filter.

    This uses `ffmpeg -filters` and looks for the `subtitles` filter in the output.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"], capture_output=True, text=True, check=True
        )
        # case-insensitive check is safer; but ffmpeg prints lower-case
        return "subtitles" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run_ffmpeg_and_log(cmd: Any, verbose: bool = False):
    """
    Run an ffmpeg-python command and print stderr stdout on failure.
    """
    try:
        # ffmpeg-python exposes `.run()` on commands; let it raise on error
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


def _run_ffmpeg_cli_and_log(args: list, cwd: str | None = None, verbose: bool = False):
    """
    Run a raw ffmpeg CLI via subprocess and log stdout/stderr on failure.
    """
    if verbose:
        print("Running ffmpeg command:", " ".join(args))
    try:
        # Use subprocess.run to control cwd and capture output
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
    """Return a safely quoted value for ffmpeg filter option parsing.

    The ffmpeg filter parser supports quoting around option values; this
    helper attempts a pragmatic, cross-platform quoting strategy:
    - Prefer single quotes unless the string contains single quotes, then
      use double quotes.
    - If both single and double quotes appear, escape single quotes.
    """
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    # Both single and double quotes present, escape single quotes
    return "'" + value.replace("'", "\'") + "'"


def add_subtitles_to_video(input_path: str, srt_path: str, out_path: str, verbose: bool = False, mode: str = "burn") -> None:
    """
    Burn subtitles into the video using ffmpeg.

    Notes
    -----
    - Normalizes Windows paths to avoid parsing problems in the ffmpeg filter.
    - Re-encodes to `libx264` and `aac` to avoid container compatibility issues.
    - Captures ffmpeg stderr on failure and prints it.
    """
    if mode == "external":
        # Just copy the input to out_path without hardcoding or embedding
        # the subtitles stream. If this fails due to container issues,
        # fall back to re-encoding.
        try:
            cmd = ffmpeg.input(input_path).output(out_path, vcodec="copy", acodec="copy")
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error:
            cmd = ffmpeg.input(input_path).output(out_path, vcodec="libx264", acodec="aac")
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        return

    if not _ffmpeg_supports_subtitles() and mode == "burn":
        print("Warning: this ffmpeg build may not support the 'subtitles' filter.")
        print("Run 'ffmpeg -filters | findstr subtitles' to verify and install a build with libass.")

    # Normalize paths early using pathlib for correctness across OSes
    input_path = str(Path(input_path).resolve())
    out_path = str(Path(out_path).resolve())
    srt_path = str(Path(srt_path).resolve())

    # Convert path to POSIX-style for the ffmpeg filter, which expects
    # forward slashes and may choke on Windows backslashes.
    normalized_srt = Path(srt_path).as_posix()
    # For debugging, print normalized srt path if verbose
    if verbose:
        print(f"Using subtitles file path for filter: {normalized_srt}")

    video = ffmpeg.input(input_path)
    audio = video.audio

    # Pass the filename argument explicitly to avoid filter parsing issues on Windows
    # Note: ffmpeg-python will construct the filter string for us; however
    # ffmpeg/libass historically has issues parsing Windows drive letters
    # and escaped colons (e.g. "C\:/..."). We try using ffmpeg-python
    # first and fall back to a CLI invocation with a temporary working
    # directory to avoid drive letter parsing, if necessary.
    # Branch by mode: burn vs embed
    if mode == "burn":
        filtered_video = video.filter(
            "subtitles", filename=normalized_srt, force_style="OutlineColour=&H40000000,BorderStyle=3"
        )
        cmd = ffmpeg.output(filtered_video, audio, out_path, vcodec="libx264", acodec="aac")
        try:
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error as e:
            # Handle Windows path parsing failures with fallback
            stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
            if "Unable to open" in stderr and ":\\" in srt_path:
                # Create a short filename copy in a temp dir
                fallback_dir = tempfile.mkdtemp()
                fallback_srt = os.path.join(fallback_dir, os.path.basename(srt_path))
                try:
                    with open(srt_path, "rb") as src, open(fallback_srt, "wb") as dst:
                        dst.write(src.read())
                    # Run ffmpeg CLI with cwd set to the temp dir and pass a relative
                    # filename so the filter string doesn't contain a drive letter.
                    filter_value = _quote_for_ffmpeg_filter(os.path.basename(fallback_srt))
                    filter_arg = f"subtitles={filter_value}"
                    if verbose:
                        print("Fallback ffmpeg filter string:", filter_arg)
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
        # Embed SRT as a subtitle track in the container. Choose a codec
        # for the subtitle stream depending on container.
        ext = Path(out_path).suffix.lower()
        if ext in [".mp4", ".mov", ".m4v"]:
            scodec = "mov_text"
        else:
            # For MKV and others, use 'subrip' (SRT) codec where supported.
            scodec = "subrip"

        srt_input = ffmpeg.input(srt_path)
        # Try to copy the existing audio/video; otherwise re-encode as fallback
        try:
            cmd = ffmpeg.output(video, srt_input, out_path, vcodec="copy", acodec="copy", scodec=scodec)
            _run_ffmpeg_and_log(cmd, verbose=verbose)
        except ffmpeg.Error:
            cmd = ffmpeg.output(video, srt_input, out_path, vcodec="libx264", acodec="aac", scodec=scodec)
            _run_ffmpeg_and_log(cmd, verbose=verbose)
    # No single final run; each branch already performed the run, so nothing
    # left to do here.


if __name__ == '__main__':
    main()
