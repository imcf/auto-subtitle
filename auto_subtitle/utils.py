import os
from typing import Iterator, TextIO, Iterable, List, Tuple
import re
import textwrap


def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _wrap_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    # First use textwrap to wrap on word boundaries
    wrapped = textwrap.wrap(text, width=max_chars)
    if len(wrapped) <= max_lines:
        return wrapped
    # If wraps exceed max lines, merge the last lines to fit into max_lines
    # by rebinding them into max_lines parts roughly equal in size.
    joined = " ".join(wrapped)
    # Split into max_lines chunks using a simple proportional split by character count
    parts = []
    total_len = len(joined)
    approx = total_len // max_lines
    start = 0
    for i in range(max_lines - 1):
        # find next whitespace after approx
        end = start + approx
        while end < total_len and joined[end] != " ":
            end += 1
        parts.append(joined[start:end].strip())
        start = end + 1
    parts.append(joined[start:].strip())
    return parts


def _split_into_chunks(text: str, start: float, end: float, max_chars: int, max_lines: int, max_duration: float, min_duration: float) -> List[Tuple[float, float, str]]:
    """Split text into multiple (start, end, text) chunks by duration and length.

    Strategy:
    - Prefer break at sentence boundaries (.,!,?)
    - If sentences are still too long, break by words so each chunk has <= max_chars * max_lines
    - Distribute timestamps proportionally by word count to maintain original timing
    - Ensure each chunk duration is >= min_duration by merging if necessary
    """
    duration = max(0.0, end - start)
    max_chars_total = max_chars * max_lines
    text = text.strip()
    if not text:
        return []

    # If text already small and under max duration, no split needed
    if duration <= max_duration and len(text) <= max_chars_total:
        return [(start, end, text)]

    # Break into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Combine sentences into chunks not exceeding max_chars_total
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if current:
            candidate = current + " " + s
        else:
            candidate = s
        if len(candidate) <= max_chars_total:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If this single sentence itself is too long, break by words
            if len(s) > max_chars_total:
                words = s.split()
                cur_words = []
                cur_len = 0
                for w in words:
                    if cur_len + len(w) + (1 if cur_words else 0) <= max_chars_total:
                        cur_words.append(w)
                        cur_len += len(w) + (1 if cur_words else 0)
                    else:
                        chunks.append(" ".join(cur_words))
                        cur_words = [w]
                        cur_len = len(w)
                if cur_words:
                    chunks.append(" ".join(cur_words))
                current = ""
            else:
                current = s
    if current:
        chunks.append(current)

    # Now break by words if still longer than max_chars_total
    final_chunks = []
    for c in chunks:
        if len(c) <= max_chars_total:
            final_chunks.append(c)
        else:
            words = c.split()
            cur = []
            cur_len = 0
            for w in words:
                if cur_len + len(w) + (1 if cur else 0) <= max_chars_total:
                    cur.append(w)
                    cur_len += len(w) + (1 if cur else 0)
                else:
                    final_chunks.append(" ".join(cur))
                    cur = [w]
                    cur_len = len(w)
            if cur:
                final_chunks.append(" ".join(cur))

    # Convert chunks to (start,end,text) using per-word duration
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        # fallback: split evenly by chars
        n = max(1, len(final_chunks))
        per_dur = duration / n if n else duration
        out = []
        cur_start = start
        for i, c in enumerate(final_chunks):
            cur_end = start + (i + 1) * per_dur
            out.append((cur_start, cur_end, c.strip()))
            cur_start = cur_end
        return out

    per_word_duration = duration / total_words if total_words > 0 else 0
    current_word_index = 0
    out = []
    for c in final_chunks:
        n_words = len(c.split())
        chunk_start = start + current_word_index * per_word_duration
        chunk_end = chunk_start + n_words * per_word_duration
        out.append((chunk_start, chunk_end, c.strip()))
        current_word_index += n_words

    # Merge very short segments with next chunk
    merged = []
    for seg in out:
        if merged and seg[1] - seg[0] < min_duration:
            # merge with previous
            prev_start, prev_end, prev_text = merged.pop()
            merged.append((prev_start, seg[1], f"{prev_text} {seg[2]}"))
        else:
            merged.append(seg)

    # Ensure min_duration by merging the last segments if needed
    if merged and merged[-1][1] - merged[-1][0] < min_duration and len(merged) > 1:
        prev_start, prev_end, prev_text = merged[-2]
        last_start, last_end, last_text = merged[-1]
        merged[-2] = (prev_start, last_end, f"{prev_text} {last_text}")
        merged.pop()

    return merged


def write_srt(transcript: Iterator[dict], file: TextIO, max_chars_per_line: int = 42, max_lines: int = 2,
              max_duration: float = 5.0, min_duration: float = 0.5):
    """Write SRT from transcript segments, with optional splitting/wrapping.

    Parameters
    ----------
    transcript : Iterator[dict]
        Iterator of speech segments containing 'start', 'end' and 'text'.
    file : TextIO
        Output file-like object.
    max_chars_per_line : int
        Maximum characters per line when wrapping.
    max_lines : int
        Maximum number of lines per subtitle.
    max_duration : float
        Maximum duration (seconds) for any subtitle segment. Longer segments are split.
    min_duration : float
        Minimum duration (seconds) allowed for a subtitle segment. Too-short segments are merged.
    """
    idx = 1
    for segment in transcript:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = segment.get("text", "").strip().replace('-->', '->')
        # Split into chunks if necessary
        chunks = _split_into_chunks(text, start, end, max_chars_per_line, max_lines, max_duration, min_duration)
        if not chunks:
            continue
        for s, e, t in chunks:
            lines = _wrap_lines(t, max_chars_per_line, max_lines)
            lines_joined = "\n".join(lines)
            print(
                f"{idx}\n"
                f"{format_timestamp(s, always_include_hours=True)} --> {format_timestamp(e, always_include_hours=True)}\n"
                f"{lines_joined}\n",
                file=file,
                flush=True,
            )
            idx += 1


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]
