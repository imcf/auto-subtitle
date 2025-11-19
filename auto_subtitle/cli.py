import os
import subprocess
import ffmpeg
import whisper
import argparse
import shlex
import shutil
import sys
import warnings
import tempfile
from typing import Dict, Any, Callable, Optional
from .ffmpeg_utils import (
    add_subtitles_to_video,
    _ffmpeg_supports_subtitles,
    _run_ffmpeg_and_log,
    _run_ffmpeg_cli_and_log,
    _quote_for_ffmpeg_filter,
)
from pathlib import Path
from .utils import filename, str2bool, write_srt


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="*", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing video files to process in batch")
    parser.add_argument("--recursive", action='store_true', default=False,
                        help="Recursively scan input_dir for video files")
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
    parser.add_argument("--edit_srt", type=str2bool, default=False,
                        help="Open generated SRT files for manual editing before burning/embedding.")
    parser.add_argument("--editor", type=str, default=None,
                        help="Editor command to use for editing SRT files. Can include args, e.g. 'code --wait'.")
    parser.add_argument("--batch", action='store_true', default=False,
                        help="After generating SRTs, run burn on all files in batch mode (non-interactive)")
    parser.add_argument("--gui", type=str2bool, default=False,
                        help="Open a local web GUI allowing subtitle editing while previewing the video")
    parser.add_argument("--gui_port", type=int, default=5000,
                        help="Port for the local GUI server")

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
    edit_srt: bool = args.pop("edit_srt")
    editor_cmd: str | None = args.pop("editor")
    gui_enabled: bool = args.pop("gui")
    gui_port: int = args.pop("gui_port")
    batch_mode: bool = args.pop("batch")
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
    audios = get_audio(videos)
    subtitles = get_subtitles(
        audios, output_srt or srt_only, output_dir, lambda audio_path: model.transcribe(audio_path, **args),
        max_chars, max_lines, max_sub_duration, min_sub_duration
    )

    if srt_only:
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

    if gui_enabled and subtitles:
        items = list(subtitles.items())
        print(f"Launching GUI on http://127.0.0.1:{gui_port} for {len(items)} files...")
        from .gui import run as run_gui
        run_gui(items, port=gui_port)
        # When GUI is used for editing/burning, do not auto-burn from the CLI
        return

    if batch_mode:
        # Batch burn all files using the requested mode: subtitle_mode
        for path, srt_path in subtitles.items():
            out_path = os.path.join(output_dir, f"{filename(path)}.mp4")
            print(f"Batch burning {path}...")
            add_subtitles_to_video(path, srt_path, out_path, verbose, mode=subtitle_mode)
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



def _open_file_in_editor(path: str, editor: Optional[str], verbose: bool = False) -> None:
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
        subprocess.run(args, check=True)
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
        subprocess.run(args, check=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, env_editor, verbose)
        return

    # Platform-specific defaults
    if sys.platform == "win32":
        # Notepad blocks until closed
        subprocess.run(["notepad", path], check=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, None, verbose)
        return
        return
    elif sys.platform == "darwin":
        # open -W waits for the app to close
        subprocess.run(["open", "-W", path], check=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, None, verbose)
        return
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
            subprocess.Popen(["xdg-open", path])
            input("Press Enter when finished editing the subtitle file to continue...")
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f"Warning: {path} is empty or missing after editing.")
                if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                    _open_file_in_editor(path, None, verbose)
            return
            return
        # last resort: print a message and return
        print(f"Couldn't find an editor; please edit {path} manually and press Enter when done.")
        input("Press Enter when finished editing the subtitle file to continue...")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Warning: {path} is empty or missing after editing.")
            if input("Re-open to edit? (y/N): ").lower() in ("y", "yes"):
                _open_file_in_editor(path, None, verbose)
        return


from .ffmpeg_utils import _ffmpeg_supports_subtitles, _run_ffmpeg_and_log, _run_ffmpeg_cli_and_log, _quote_for_ffmpeg_filter


# add_subtitles_to_video pulled from ffmpeg_utils, use that helper instead
    # No single final run; each branch already performed the run, so nothing
    # left to do here.


if __name__ == '__main__':
    main()
