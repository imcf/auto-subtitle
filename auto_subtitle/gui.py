from __future__ import annotations

import os
import threading
import webbrowser
from typing import Optional, List, Tuple, Dict
from .ffmpeg_utils import add_subtitles_to_video

from flask import Flask, request, send_file, jsonify, render_template_string

from .utils import filename


def _srt_to_vtt(srt_content: str) -> str:
    """Convert SRT text to WebVTT text for browser subtitle tracks.

    Basic conversion: add 'WEBVTT' header and replace time commas with dots.
    """
    lines = srt_content.splitlines()
    out_lines = ["WEBVTT\n"]
    for line in lines:
        if "-->" in line:
            # Replace comma decimal separator with dot
            line = line.replace(",", ".")
        # skip numeric indexes (single number lines) to avoid being interpreted
        out_lines.append(line)
    return "\n".join(out_lines)


def _create_app(items: List[Tuple[str, str]]) -> Flask:
    """Create Flask app for multiple video/srt items.

    items: list of (video_path, srt_path)
    """
    app = Flask(__name__)
    # store items in app context
    app.config["ITEMS"] = items
    # status map for burning
    app.config["STATUS"] = {i: {"state": "idle", "msg": ""} for i in range(len(items))}

    @app.route("/")
    def index():
        video_name = os.path.basename(items[0][0]) if items else ""
        return render_template_string(
            _INDEX_HTML,
            video_name=video_name,
        )

    @app.route("/list")
    def list_items():
        items = app.config["ITEMS"]
        return jsonify([{"index": i, "video": os.path.basename(v), "video_path": v, "srt_path": s} for i, (v, s) in enumerate(items)])

    @app.route("/video/<int:idx>")
    def video(idx: int):
        items = app.config["ITEMS"]
        if idx < 0 or idx >= len(items):
            return "", 404
        return send_file(items[idx][0])

    @app.route("/srt/<int:idx>")
    def srt(idx: int):
        items = app.config["ITEMS"]
        if idx < 0 or idx >= len(items):
            return "", 404
        srt_path = items[idx][1]
        if not os.path.exists(srt_path):
            return "", 404
        return send_file(srt_path)

    @app.route("/srt.vtt/<int:idx>")
    def srt_vtt(idx: int):
        items = app.config["ITEMS"]
        if idx < 0 or idx >= len(items):
            return "", 404
        srt_path = items[idx][1]
        if not os.path.exists(srt_path):
            return "", 404
        with open(srt_path, "r", encoding="utf-8") as fh:
            srt_text = fh.read()
        vtt = _srt_to_vtt(srt_text)
        return app.response_class(vtt, mimetype="text/vtt")

    @app.route("/save_srt/<int:idx>", methods=["POST"])
    def save_srt(idx: int):
        items = app.config["ITEMS"]
        if idx < 0 or idx >= len(items):
            return jsonify({"ok": False, "error": "Index out of range"}), 400
        content = request.form.get("content")
        if content is None:
            return jsonify({"ok": False, "error": "No content provided"}), 400
        srt_path = items[idx][1]
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return jsonify({"ok": True})

    @app.route("/status")
    def status():
        return jsonify(app.config.get("STATUS", {}))

    @app.route("/burn", methods=["POST"])
    def burn():
        payload = request.get_json() or {}
        idx = payload.get("idx")
        mode = payload.get("mode", "burn")
        # when idx is 'all' or None => burn all
        items = app.config["ITEMS"]
        status = app.config["STATUS"]

        def _burn_one(i):
            vpath, spath = items[i]
            status[i] = {"state": "running", "msg": ""}
            try:
                add_subtitles_to_video(vpath, spath, vpath.rsplit('.',1)[0] + ".subbed.mp4", verbose=True, mode=mode)
                status[i] = {"state": "success", "msg": ""}
            except Exception as e:
                status[i] = {"state": "failed", "msg": str(e)}

        def _burn_all():
            for i in range(len(items)):
                _burn_one(i)

        if idx is None or idx == "all":
            threading.Thread(target=_burn_all, daemon=True).start()
            return jsonify({"ok": True, "queued": "all"})
        try:
            idx = int(idx)
        except Exception:
            return jsonify({"ok": False, "error": "Invalid idx"}), 400
        if idx < 0 or idx >= len(items):
            return jsonify({"ok": False, "error": "Index out of range"}), 400
        threading.Thread(target=_burn_one, args=(idx,), daemon=True).start()
        return jsonify({"ok": True, "queued": idx})

    @app.route("/done", methods=["POST"])
    def done():
        # shutdown the server
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            return "Server shutdown not available", 500
        func()
        return "OK"

    return app


def run(items: List[Tuple[str, str]], port: int = 5000, host: str = "127.0.0.1") -> None:
    """Start the Flask GUI for video + subtitle editing and block until done.

    This function will launch a development server bound to host:port and open
    the user's web browser to the page. It blocks until the 'Done' button from
    the page shuts down the server.
    """
    app = _create_app(items)
    url = f"http://{host}:{port}/"

    # Start server in a background thread
    server = threading.Thread(target=lambda: app.run(host=host, port=port, threaded=True), daemon=True)
    server.start()

    # open URL in browser
    webbrowser.open(url)

    # wait for thread to finish
    server.join()


_INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Subtitle Editor</title>
    <style>
      body { font-family: sans-serif; padding: 1rem; display: flex; gap: 1rem; }
      .left { flex: 2; }
      .right { flex: 1; display: flex; flex-direction: column; }
      textarea { width: 100%; height: 80vh; font-family: monospace; }
      video { width: 100%; height: auto; border: 1px solid #ccc; }
      button { margin-top: 0.5rem; }
    </style>
  </head>
  <body>
    <div class="left">
      <div style="display:flex; gap: 0.5rem; align-items:center;">
        <button id="prev">Prev</button>
        <video id="video" controls crossorigin>
          <source id="video-src" src="/video/0" type="video/mp4">
          <track id="track" label="Subtitles" kind="subtitles" srclang="en" src="/srt.vtt/0" default>
        </video>
        <button id="next">Next</button>
      </div>
      <div style="margin-top:0.5rem; display:flex; gap: 0.5rem; align-items:center;">
        <select id="file-list"></select>
        <label for="subtitle-mode">Mode:</label>
        <select id="subtitle-mode">
          <option value="burn">Burn</option>
          <option value="embed">Embed</option>
          <option value="external">External</option>
        </select>
      </div>
    </div>
    <div class="right">
      <h3>{{ video_name }}</h3>
      <textarea id="srt-area"></textarea>
      <div style="display:flex; gap:0.5rem; align-items:center;">
        <button id="save">Save</button>
        <button id="reload">Reload Subtitles</button>
        <button id="burn">Burn This</button>
        <button id="burnAll">Burn All</button>
        <button id="done">Done</button>
      </div>
      <div style="margin-top:0.5rem; color: #666;">Status: <span id="status">idle</span></div>
      <div style="margin-top:0.5rem; color: #666;">Edit the subtitles and click Save. Use Reload Subtitles to reload into the player.</div>
    </div>
    <script>
      let currentIndex = 0;
      async function loadList() {
        const res = await fetch('/list');
        const items = await res.json();
        const select = document.getElementById('file-list');
        select.innerHTML = '';
        items.forEach(it => {
          const opt = document.createElement('option');
          opt.value = it.index;
          opt.textContent = it.video;
          select.appendChild(opt);
        });
      }

      async function loadSrt(index = 0) {
        const res = await fetch('/srt/' + index);
        if (!res.ok) { document.getElementById('srt-area').value = ''; return; }
        const text = await res.text();
        document.getElementById('srt-area').value = text;
      }
      document.getElementById('save').addEventListener('click', async () => {
        const body = new URLSearchParams();
        body.append('content', document.getElementById('srt-area').value);
        const res = await fetch('/save_srt/' + currentIndex, { method: 'POST', body });
        if (res.ok) { alert('Saved.'); } else { alert('Failed to save.'); }
      });
      document.getElementById('reload').addEventListener('click', async () => {
        // reload the track, append cache buster
        const track = document.getElementById('track');
        track.src = '/srt.vtt/' + currentIndex + '?ts=' + Date.now();
        track.mode = 'showing';
        const v = document.getElementById('video');
        v.load();
      });
      document.getElementById('prev').addEventListener('click', async () => { selectIndex(currentIndex - 1); });
      document.getElementById('next').addEventListener('click', async () => { selectIndex(currentIndex + 1); });
      document.getElementById('file-list').addEventListener('change', async (e) => { selectIndex(parseInt(e.target.value)); });
      document.getElementById('burn').addEventListener('click', async () => { queueBurn(currentIndex); });
      document.getElementById('burnAll').addEventListener('click', async () => { queueBurn('all'); });
      async function selectIndex(idx) {
        const res = await fetch('/list');
        const items = await res.json();
        if (idx < 0) idx = 0;
        if (idx >= items.length) idx = items.length - 1;
        currentIndex = idx;
        document.getElementById('file-list').value = idx;
        document.getElementById('video-src').src = '/video/' + idx;
        const track = document.getElementById('track');
        track.src = '/srt.vtt/' + idx + '?ts=' + Date.now();
        const v = document.getElementById('video');
        v.load();
        await loadSrt(idx);
        updateStatus();
      }
      async function queueBurn(idx) {
        document.getElementById('status').innerText = 'queued';
        const mode = document.getElementById('subtitle-mode').value || 'burn';
        const res = await fetch('/burn', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({idx: idx, mode: mode})});
        const j = await res.json();
        if (!j.ok) alert('Failed to queue burn');
        updateStatus();
      }
      async function updateStatus() {
        const res = await fetch('/status');
        const st = await res.json();
        const s = st[currentIndex] || {state:'idle'};
        document.getElementById('status').innerText = s.state + (s.msg ? ': ' + s.msg : '');
      }
      document.getElementById('done').addEventListener('click', async () => {
        await fetch('/done', { method: 'POST' });
        alert('GUI closed.');
      });
      loadList().then(() => selectIndex(0));
      setInterval(updateStatus, 2000);
    </script>
  </body>
</html>
"""
