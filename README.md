# Automatic subtitles in your videos

This repository uses `ffmpeg` and [OpenAI's Whisper](https://openai.com/blog/whisper) to automatically generate and overlay subtitles on any video.

## Installation

To get started, you'll need Python 3.7 or newer. Install the binary by running the following command:

    pip install git+https://github.com/m1guelpf/auto-subtitle.git

You'll also need to install [`ffmpeg`](https://ffmpeg.org/), which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg
```

## Usage

The following command will generate a `subtitled/video.mp4` file contained the input video with overlayed subtitles.

    auto_subtitle /path/to/video.mp4 -o subtitled/

The default setting (which selects the `small` model) works well for transcribing English. You can optionally use a bigger model for better results (especially with other languages). The available models are `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`.

    auto_subtitle /path/to/video.mp4 --model medium

Adding `--task translate` will translate the subtitles into English:

    auto_subtitle /path/to/video.mp4 --task translate

Run the following to view all available options:

    auto_subtitle --help

## New options & workflows

This project gained several new options and a simple web GUI that makes editing and batch processing easier.

Key options (simple explanation)

- `--input_dir DIR` — process all videos in a folder (common extensions: mp4, mov, mkv, avi, webm, mpg). See `--recursive` to include subfolders.
- `--recursive` — when used with `--input_dir`, recursively scan the directory tree for videos.
- `--output_dir, -o DIR` — where to write output files (videos & SRTs). Defaults to the current directory.
- `--output_srt True` — write SRT files in `output_dir` besides the video files.
- `--srt_only True` — only write SRT files; do not burn/encode video outputs.
- `--subtitle_mode [burn|embed|external]` —
    - `burn` — draw (hardcode) subtitles directly onto the video (default)
    - `embed` — add a selectable subtitle track to the output container (mov_text or subrip)
    - `external` — do not add or burn any subtitles; just write the `.srt` file
- `--edit_srt True` — opens generated SRT files in your editor before burning/embedding, allowing manual corrections.
- `--editor "command --args"` — customize editor command if you want to use VS Code or other editors (example: `--editor "code --wait"`).
- `--gui True` — launch an interactive web GUI for previewing video(s) and editing subtitles live. Supports batch lists and burning of single/all files from the UI.
- `--gui_port PORT` — specify the port for the GUI server (default: 5000).
- `--batch True` — after generating SRTs, run the burn process on all videos in non-interactive batch mode.

Subtitle formatting & split behavior

- `--max_chars_per_line N` — maximum characters per subtitle line (word wrapped), default 42.
- `--max_lines N` — allow up to N lines per subtitle block, default 2.
- `--max_sub_duration X` — maximum seconds for subtitle duration; longer segments are split (default 5.0).
- `--min_sub_duration X` — minimum seconds for a subtitle entry; too-short segments are merged with neighbors (default 0.5).

Examples

- Process a single file and burn subtitles (default behavior):
    ```powershell
    auto_subtitle "C:\videos\file.mp4" -o out_dir
    ```

- Process a directory recursively and open the GUI for preview/editing:
    ```powershell
    auto_subtitle --input_dir "C:\videos" --recursive --gui True
    ```

- Generate only SRTs for all videos and open them in your editor for manual corrections (VS Code example):
    ```powershell
    auto_subtitle --input_dir "C:\videos" --recursive --srt_only True --edit_srt True --editor "code --wait"
    ```

- Generate SRTs for all files and automatically burn them (non-interactive):
    ```powershell
    auto_subtitle --input_dir "C:\videos" --recursive --batch True --subtitle_mode burn
    ```

- Embed subtitles rather than burning them:
    ```powershell
    auto_subtitle "file.mp4" --subtitle_mode embed -o out_dir
    ```

- Notes & Tips

- If you plan to edit subtitles interactively, use `--gui` — it supports a dropdown to pick any generated file, an editor textarea, Save/Reload, and buttons to Burn This / Burn All.
- If you use VS Code and want it to block until you close the file, set `--editor "code --wait"`.
- Installing Flask enables the GUI. If Flask is not installed and you pass `--gui`, the CLI will show a helpful error.
- Ensure `ffmpeg` is installed (and includes libass if you need to burn with the `subtitles` filter). If you see errors such as "Unable to open C\:/...", use the GUI and try saving the SRT to a path without drive colons or test a non-Windows path style.

Advanced: If you prefer the behavior of copying codecs, or tweaking the final naming convention for `subbed.mp4` outputs, let me know and I can add the optional flags for these.

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
