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

This project gained several new options to improve editing and batch processing.

Key options (simple explanation)

- `--input_dir DIR` — process all videos in a folder (common extensions: mp4, mov, mkv, avi, webm, mpg). See `--recursive` to include subfolders.
- `--recursive` — when used with `--input_dir`, recursively scan the directory tree for videos.
- `--output_dir, -o DIR` — where to write output files (videos). Defaults to the current directory. SRT files are saved next to the original video files when `--output_srt True`.
- `--output_srt True` — write SRT files next to the original video files (same directory), using the same basename as the video (e.g., `video.mp4` → `video.srt`).
- `--srt_only True` — only write SRT files; do not burn/encode video outputs.
- `--subtitle_mode [burn|embed|external]` —
    - `burn` — draw (hardcode) subtitles directly onto the video (default)
    - `embed` — add a selectable subtitle track to the output container (mov_text or subrip)
    - `external` — do not add or burn any subtitles; just write the `.srt` file
-- `--edit_srt True` — opens generated SRT files in your editor before burning/embedding, allowing manual corrections.
-- `--editor "command --args"` — customize editor command if you want to use VS Code or other editors (example: `--editor "code --wait"`).
- `--batch True` — after generating SRTs, run the burn process on all videos in non-interactive batch mode.
- `--srt_path PATH` — path to an existing `.srt` file or a directory containing `.srt` files to use instead of generating new ones. When a directory is provided, filenames should match the video basenames (e.g., `video.mp4` -> `video.srt`).

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

- Process a directory recursively:
    ```powershell
    auto_subtitle --input_dir "C:\videos" --recursive
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

If you plan to edit subtitles interactively, use `--edit_srt` — this opens the generated SRT file in your preferred editor for manual correction before you burn or embed.
- If you use VS Code and want it to block until you close the file, set `--editor "code --wait"`.
If you use Visual Studio Code and want it to block until you close the file, set `--editor "code --wait"`.
Note: To use `code` from the command line, open VS Code and run the "Install 'code' command in PATH" command from the Command Palette (or use the Shell Command menu and follow platform instructions). Alternatively set the `EDITOR` environment variable to a GUI editor or terminal editor you have installed (e.g., `setx EDITOR "notepad"` on Windows or `export EDITOR="nano"` on Linux/macOS). The CLI will fall back to system defaults if your chosen editor command is not found.

- Ensure `ffmpeg` is installed (and includes libass if you need to burn with the `subtitles` filter).
    If you see errors such as "Unable to open C\:/..." on Windows, try using `--edit_srt` and save the SRT to a path without drive colons, or instead use `--srt_path` to point to a folder with SRT files (the CLI can also accept a single `.srt` file with `--srt_path` for single-input runs). The CLI also provides a fallback that runs `ffmpeg` from a temporary directory when path issues are detected.

### Editor workflow

When you use `--edit_srt`, the tool will open the generated `.srt` file in the editor defined by (in order):

- the `--editor` argument (explicit command, e.g. `--editor "code --wait"`)
- the `EDITOR` or `VISUAL` environment variable
- platform defaults: on Windows it launches `notepad`, on macOS it uses `open -W`, and on Linux it prefers `nano/vi/vim` or falls back to `xdg-open`.

If you want an editor that blocks (so the CLI waits), set `--editor` to something like `code --wait` for Visual Studio Code. If a specified editor command does not exist on your PATH, the CLI will print a helpful error and exit.

### Troubleshooting

- ffmpeg "Unable to open C\:/..." on Windows: this is commonly due to the subtitles filter or the path format; try moving the SRT file to a simple path (no drive colon) or run with `--output_srt` to write the file to a chosen location. The CLI also falls back to invoking `ffmpeg` from a temp directory in some cases.
- Make sure your `ffmpeg` build includes libass if you want to burn subtitles with the `subtitles` filter (required for `--subtitle_mode burn`). If you get errors about the `subtitles` filter being missing, install a full `ffmpeg` build or use `--subtitle_mode embed` / `external` as an alternative.
- For embedding rather than burning, `--subtitle_mode embed` creates a track that is selectable in players that support subtitle tracks (e.g., HTML5 players, VLC).

### Burning an existing SRT file to a video

If you already have an `.srt` file and just want to burn it into a video (hardcode the subtitles), you can do this in two ways:

You can also use the `auto_subtitle` CLI directly to burn an existing `.srt` file into a video. Examples:

- Single file + SRT: (burns `subtitles.srt` into `input.mp4` and writes to `out_dir`)
```powershell
auto_subtitle "C:\videos\input.mp4" --srt_path "C:\videos\subtitles.srt" -o "C:\videos\out_dir" --subtitle_mode burn
```

- Directory of videos + directory of .srt files: (expects `video.mp4` and `video.srt` to have same basename)
```powershell
auto_subtitle --input_dir "C:\videos" --srt_path "C:\videos\srts" --recursive --batch True --subtitle_mode burn
```

### Testing & Development

Unit tests use `pytest` and are included under `tests/`. You can run them locally after installing `pytest` (already included in `requirements.txt` for convenience):

    pip install -r requirements.txt
    python -m pytest -q

Advanced: If you prefer the behavior of copying codecs, or tweaking the final naming convention for `subbed.mp4` outputs, let me know and I can add the optional flags for these.

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.
