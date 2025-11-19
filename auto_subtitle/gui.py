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
    # Add a shutdown event to signal the run() method
    app.config["SHUTDOWN_EVENT"] = threading.Event()

    @app.route("/")
    def index():
        video_name = os.path.basename(items[0][0]) if items else ""
        if items:
          try:
            s0 = items[0][1]
            print(f"[GUI] index: video={items[0][0]} srt_exists={os.path.exists(s0)} srt_len={(os.path.getsize(s0) if os.path.exists(s0) else 0)}")
          except Exception as e:
            print(f"[GUI] index: error checking srt: {e}")
        return render_template_string(
            _INDEX_HTML,
            video_name=video_name,
        )

    @app.route("/list")
    def list_items():
      items = app.config["ITEMS"]
      # debug log out which items and whether they have srt
      print('[GUI] list_items: ' + ', '.join([f"{i}:{os.path.basename(v)} has_srt={os.path.exists(s)}" for i, (v, s) in enumerate(items)]))
      return jsonify([{
        "index": i,
        "video": os.path.basename(v),
        "video_path": v,
        "srt_path": s,
        "has_srt": os.path.exists(s),
      } for i, (v, s) in enumerate(items)])

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
          print(f"[GUI] SRT not found for idx={idx} path={srt_path}")
          return "[No subtitles found]", 404
        try:
          with open(srt_path, 'r', encoding='utf-8', errors='ignore') as fh:
            txt = fh.read()
          print(f"[GUI] Served SRT idx={idx} len={len(txt)} path={srt_path}")
          return app.response_class(txt, mimetype='text/plain')
        except Exception as e:
          print(f"[GUI] Error reading SRT idx={idx} path={srt_path} err={e}")
          return str(e), 500

    @app.route("/srt.vtt/<int:idx>")
    def srt_vtt(idx: int):
        items = app.config["ITEMS"]
        if idx < 0 or idx >= len(items):
            return "", 404
        srt_path = items[idx][1]
        if not os.path.exists(srt_path):
          print(f"[GUI] VTT requested but SRT not found idx={idx} path={srt_path}")
          return "[No subtitles found]", 404
        try:
          with open(srt_path, "r", encoding="utf-8", errors='ignore') as fh:
            srt_text = fh.read()
        except Exception as e:
          print(f"[GUI] Error reading SRT idx={idx} path={srt_path} err={e}")
          return str(e), 500
        vtt = _srt_to_vtt(srt_text)
        print(f"[GUI] Served VTT idx={idx} len={len(vtt)}")
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
        print(f"[GUI] Saved SRT for idx={idx} path={srt_path}")
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
                print(f"[GUI] Burning idx={i} vpath={vpath} srt={spath} mode={mode}")
                add_subtitles_to_video(vpath, spath, vpath.rsplit('.',1)[0] + ".subbed.mp4", verbose=True, mode=mode)
                status[i] = {"state": "success", "msg": ""}
            except Exception as e:
                print(f"[GUI] Burn failed idx={i} err={e}")
                status[i] = {"state": "failed", "msg": str(e)}

        def _burn_all():
            for i in range(len(items)):
                _burn_one(i)

        if idx is None or idx == "all":
            print("[GUI] Queued burn for all items")
            threading.Thread(target=_burn_all, daemon=True).start()
            return jsonify({"ok": True, "queued": "all"})
        try:
            idx = int(idx)
        except Exception:
            return jsonify({"ok": False, "error": "Invalid idx"}), 400
        if idx < 0 or idx >= len(items):
            return jsonify({"ok": False, "error": "Index out of range"}), 400
        print(f"[GUI] Queued burn for idx={idx}")
        threading.Thread(target=_burn_one, args=(idx,), daemon=True).start()
        return jsonify({"ok": True, "queued": idx})

    @app.route("/done", methods=["POST"])
    def done():
      # Indicate the GUI session should end; set event and attempt
      # to cleanly shut down the werkzeug server if available
      try:
        app.config["SHUTDOWN_EVENT"].set()
      except Exception:
        pass
      func = request.environ.get("werkzeug.server.shutdown")
      try:
        if func is not None:
          func()
          return jsonify({"ok": True, "shutdown": "werkzeug"})
      except Exception:
        pass
      print("[GUI] Done requested, setting event and attempting shutdown")
      # Also launch a safety thread to force exit after a short grace period
      def _force_exit():
        import time, os
        time.sleep(1.0)
        try:
          os._exit(0)
        except Exception:
          pass
      threading.Thread(target=_force_exit, daemon=True).start()
      # Emits OK and let caller close; fallback will force exit from run()
      return jsonify({"ok": True, "shutdown": "event"})

    return app


def run(items: List[Tuple[str, str]], port: int = 5000, host: str = "127.0.0.1") -> None:
    """Start the Flask GUI for video + subtitle editing and block until done.

    This function will launch a development server bound to host:port and open
    the user's web browser to the page. It blocks until the 'Done' button from
    the page shuts down the server.
    """
    app = _create_app(items)
    url = f"http://{host}:{port}/"

    # Run server in the main thread so Ctrl+C stops it in dev mode
    try:
        webbrowser.open(url)
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print('[GUI] KeyboardInterrupt received, shutting down')
    finally:
        # If the 'Done' endpoint was used, the SHUTDOWN_EVENT may have been set
        # but Werkzeug may not always shut down the process; be tidy.
        shutdown_event = app.config.get("SHUTDOWN_EVENT")
        if shutdown_event and shutdown_event.is_set():
            try:
                os._exit(0)
            except Exception:
                pass


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
      <div style="display:flex; gap: 8px; align-items:center; margin-bottom: 0.5rem;">
        <label><input type="checkbox" id="auto-reload" checked> Auto-reload after save</label>
        <label><input type="checkbox" id="sync-video-to-subs" checked> Video → Subtitles</label>
        <label><input type="checkbox" id="sync-subs-to-video" checked> Subtitles → Video</label>
      </div>
      <textarea id="srt-area"></textarea>
      <div id="segments-list" style="height: 35vh; overflow:auto; margin-top: 0.5rem; border: 1px solid #eee; padding: 0.5rem; background: #fafafa;"></div>
      <div style="display:flex; gap:0.5rem; align-items:center;">
        <button id="save">Save</button>
        <button id="reload">Reload Subtitles</button>
        <button id="fetchSrt">Fetch SRT</button>
        <button id="burn">Burn This</button>
        <button id="burnAll">Burn All</button>
        <button id="done">Done</button>
      </div>
      <label><input id="show-srt-raw" type="checkbox"> Show raw SRT</label>
      <label style="margin-left:0.5rem;"><input id="seg-edit-toggle" type="checkbox"> Edit segments inline (hide raw editor)</label>
      <pre id="srt-raw" style="display:none;white-space:pre-wrap;border:1px solid #eee;padding:0.5rem;margin-top:0.5rem;max-height:10vh;overflow:auto;font-family:monospace;background:#fff"></pre>
      <div style="margin-top:0.5rem; color: #666;">Status: <span id="status">idle</span></div>
      <div style="margin-top:0.5rem; color: #666;">Edit the subtitles and click Save. Use Reload Subtitles to reload into the player.</div>
    </div>
    <script>
      console.log('[GUI] script loaded');
      let currentIndex = 0;
      document.addEventListener('DOMContentLoaded', () => {
        async function loadList() {
        console.log('loadList start');
        try {
        const res = await fetch('/list');
        const items = await res.json();
        const select = document.getElementById('file-list');
        select.innerHTML = '';
        items.forEach(it => {
          const opt = document.createElement('option');
          opt.value = it.index;
          opt.textContent = it.video + (it.has_srt ? '' : ' (no srt)');
          select.appendChild(opt);
        });
        } catch (err) {
          console.error('Failed to load list', err);
          document.getElementById('status').innerText = 'failed: loadList';
        }
      }

      async function loadSrt(index = 0) {
        try {
          const res = await fetch('/srt/' + index);
          console.log('loadSrt status', res.status, res.statusText);
          if (!res.ok) { document.getElementById('srt-area').value = '[No subtitles found for this item]'; return; }
          const text = await res.text();
          console.log('loadSrt got text length', text.length);
          document.getElementById('srt-area').value = text;
          const pre = document.getElementById('srt-raw');
          if (pre) { pre.textContent = text; }
          if ((text || '').trim().length === 0 && (text || '').length > 0) {
            // SRT file appears to contain only whitespace; show message and the raw text
            document.getElementById('status').innerText = 'SRT contains only whitespace';
          } else {
            document.getElementById('status').innerText = 'loaded srt len: ' + text.length;
          }
        } catch (err) {
          console.error('Failed to load srt', err);
          document.getElementById('srt-area').value = '[Error loading subtitles]';
        }
      }
      document.getElementById('save').addEventListener('click', async () => {
        const body = new URLSearchParams();
        body.append('content', document.getElementById('srt-area').value);
        const res = await fetch('/save_srt/' + currentIndex, { method: 'POST', body });
        if (res.ok) { alert('Saved.'); } else { alert('Failed to save.'); }
        // auto-reload if enabled
        if (document.getElementById('auto-reload').checked) {
          await loadSrt(currentIndex);
          document.getElementById('reload').click();
        }
      });
      document.getElementById('show-srt-raw').addEventListener('change', (e) => {
        const pre = document.getElementById('srt-raw');
        pre.style.display = e.target.checked ? 'block' : 'none';
      });
      document.getElementById('seg-edit-toggle').addEventListener('change', (e) => {
        const srtArea = document.getElementById('srt-area');
        const segList = document.getElementById('segments-list');
        if (e.target.checked) {
          srtArea.style.display = 'none';
          segList.style.display = 'block';
        } else {
          srtArea.style.display = 'block';
          segList.style.display = 'block';
        }
      });

      document.getElementById('reload').addEventListener('click', async () => {
        // reload the track, append cache buster
        const track = document.getElementById('track');
        track.src = '/srt.vtt/' + currentIndex + '?ts=' + Date.now();
        track.mode = 'showing';
        const v = document.getElementById('video');
        v.load();
        // re-parse segments and render list
        parseAndRenderSegments();
      });
      document.getElementById('fetchSrt').addEventListener('click', async () => {
        await loadSrt(currentIndex);
        parseAndRenderSegments();
      });
      document.getElementById('prev').addEventListener('click', async () => { selectIndex(currentIndex - 1); });
      document.getElementById('next').addEventListener('click', async () => { selectIndex(currentIndex + 1); });
      document.getElementById('file-list').addEventListener('change', async (e) => { selectIndex(parseInt(e.target.value)); });
      document.getElementById('burn').addEventListener('click', async () => { queueBurn(currentIndex); });
      document.getElementById('burnAll').addEventListener('click', async () => { queueBurn('all'); });
      async function selectIndex(idx) {
        console.log('selectIndex start', idx);
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
        const item = items[idx] || {};
        const saveBtn = document.getElementById('save');
        const burnBtn = document.getElementById('burn');
        const burnAllBtn = document.getElementById('burnAll');
        if (!item.has_srt) {
          document.getElementById('srt-area').value = '[No subtitles found for this item]';
          saveBtn.disabled = true;
          burnBtn.disabled = true;
        } else {
          saveBtn.disabled = false;
          burnBtn.disabled = false;
          await loadSrt(idx);
          console.log('selectIndex loaded srt len', document.getElementById('srt-area').value.length);
        }
        // burnAll can still be used when some entries have SRTs
        burnAllBtn.disabled = false;

        parseAndRenderSegments();
        console.log('selectIndex parsedSegments length', parsedSegments.length);
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

      function parseSrt(srtText) {
        // parse SRT into segments [{start,end,text,raw,startChar}]
        const blocks = srtText.split(/\\r?\\n\\s*\\r?\\n/);
        const segments = [];
        let cursor = 0;
        for (let b of blocks) {
          const trimmed = b.trim();
          if (!trimmed) { cursor += b.length + 2; continue; }
          const lines = trimmed.split("\\n");
          // find time line (second line typically)
          let timeLineIndex = -1;
          for (let i=0;i<lines.length;i++){
            if (lines[i].includes("-->")) { timeLineIndex = i; break; }
          }
          if (timeLineIndex < 0) { cursor += b.length + 2; continue; }
          const timeLine = lines[timeLineIndex];
          const m = timeLine.match(/(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})/);
          if (!m) { cursor += b.length + 2; continue; }
          function parseTime(ts) {
            const parts = ts.split(":");
            const hours = parseInt(parts[0]);
            const minutes = parseInt(parts[1]);
            const secParts = parts[2].split(",");
            const seconds = parseInt(secParts[0]);
            const ms = parseInt(secParts[1]);
            return hours*3600 + minutes*60 + seconds + ms/1000.0;
          }
          const start = parseTime(m[1]);
          const end = parseTime(m[2]);
          // compute startChar index in original text via content search starting at cursor
          const findIdx = srtText.indexOf(trimmed, cursor);
          const startChar = findIdx >= 0 ? findIdx : cursor;
          segments.push({start, end, text: lines.slice(timeLineIndex+1).join("\\n"), raw: trimmed, startChar});
          cursor = startChar + trimmed.length + 2;
        }
        return segments;
      }

      function renderSegmentsList(segments) {
        const list = document.getElementById('segments-list');
        list.innerHTML = '';
        segments.forEach((seg, i) => {
          const el = document.createElement('div');
          el.className = 'segment';
          el.style.padding = '6px';
          el.style.borderBottom = '1px solid #eee';
          el.style.cursor = 'pointer';
          el.dataset.index = i;
          el.innerHTML = `<strong>${formatTime(seg.start)} - ${formatTime(seg.end)}</strong><div>${escapeHtml(seg.text)}</div>`;
          el.addEventListener('click', () => {
            if (document.getElementById('sync-subs-to-video').checked) {
              const v = document.getElementById('video');
              v.currentTime = seg.start + 0.01;
            }
            // scroll raw textarea
            const ta = document.getElementById('srt-area');
            ta.selectionStart = seg.startChar;
            ta.selectionEnd = seg.startChar + seg.raw.length;
            ta.focus();
          });
          // double click toggles inline edit
          el.addEventListener('dblclick', async () => {
            const editToggle = document.getElementById('seg-edit-toggle');
            if (!editToggle.checked) return;
            // create an editable textarea for this segment's text only
            const contentDiv = el.querySelector('div');
            const oldText = seg.text;
            const ta = document.createElement('textarea');
            ta.value = oldText;
            ta.style.width = '100%';
            ta.style.boxSizing = 'border-box';
            contentDiv.replaceWith(ta);
            ta.focus();
            // commit on blur
            ta.addEventListener('blur', () => {
              const newText = ta.value;
              // update internal parsedSegments
              parsedSegments[i].text = newText;
              // rebuild SRT from segments, update srt-area and srt-raw, re-render list
              rebuildSrtFromSegments();
            });
          });
          list.appendChild(el);
        });
      }

      function formatTime(t) {
        const h = Math.floor(t/3600);
        const m = Math.floor((t%3600)/60);
        const s = Math.floor(t%60);
        const ms = Math.floor((t - Math.floor(t)) * 1000);
        return (h>0?String(h).padStart(2,'0')+':':'') + String(m).padStart(2,'0')+ ':' + String(s).padStart(2,'0') + ',' + String(ms).padStart(3,'0');
      }

      function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\\n/g,'<br/>');
      }

      let parsedSegments = [];

      function parseAndRenderSegments() {
        const ta = document.getElementById('srt-area');
        try {
          parsedSegments = parseSrt(ta.value);
          renderSegmentsList(parsedSegments);
        } catch (err) {
          console.error('Failed to parse SRT:', err);
          document.getElementById('status').innerText = 'parse error';
        }
      }

      function rebuildSrtFromSegments() {
        // Reconstruct SRT text from parsedSegments array using the original timing
        const lines = parsedSegments.map((seg, idx) => {
          // Check for original index/number line in seg.raw
          const rawLines = seg.raw.split(/\\r?\\n/);
          let idxLine = null;
          let timeLine = null;
          if (rawLines.length > 1 && /^\s*\d+\s*$/.test(rawLines[0])) {
            idxLine = rawLines[0];
            timeLine = rawLines[1];
          } else {
            timeLine = rawLines[0].includes('-->') ? rawLines[0] : rawLines.find(l => l.includes('-->'));
          }
          const text = (seg.text || '').replace(/\\r?\\n/g, '\\\\n');
          const textLines = String(text).split('\\n');
          const blockLines = [];
          if (idxLine) blockLines.push(idxLine);
          if (timeLine) blockLines.push(timeLine);
          blockLines.push(...textLines);
          return blockLines.join('\n');
        });
        const newSrt = lines.join('\n\n');
        const srtArea = document.getElementById('srt-area');
        srtArea.value = newSrt;
        const pre = document.getElementById('srt-raw');
        if (pre) pre.textContent = newSrt;
        // Refresh the VTT track from server by writing to file: we just updated textarea
        // If auto-reload is checked, reload vtt
        if (document.getElementById('auto-reload').checked) {
          const track = document.getElementById('track');
          track.src = '/srt.vtt/' + currentIndex + '?ts=' + Date.now();
          document.getElementById('video').load();
        }
        // Re-parse to update segments' raw and startChar if necessary, then re-render
        try {
          parsedSegments = parseSrt(newSrt);
          renderSegmentsList(parsedSegments);
        } catch (err) {
          console.error('rebuildSrtFromSegments parse error', err);
        }
      }

      // Global error handler: show error in UI and console
      window.onerror = function (message, source, lineno, colno, error) {
        console.error('GUI error', message, source, lineno, colno, error);
        try { document.getElementById('status').innerText = 'error: ' + message; } catch (e) {}
        return false;
      };

      // highlight segment as video plays
      document.getElementById('video').addEventListener('timeupdate', () => {
        if (!document.getElementById('sync-video-to-subs').checked) return;
        const t = document.getElementById('video').currentTime;
        let idx = -1;
        for (let i=0;i<parsedSegments.length;i++) {
          if (t >= parsedSegments[i].start && t <= parsedSegments[i].end) { idx = i; break; }
        }
        // update selected class
        const nodes = document.getElementById('segments-list').children;
        for (let i=0;i<nodes.length;i++) {
          nodes[i].style.background = (i===idx) ? '#ffe' : 'transparent';
        }
        if (idx >= 0) {
          // scroll list to item
          const el = nodes[idx];
          if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'});
          // scroll textarea to selection
          const ta = document.getElementById('srt-area');
          const seg = parsedSegments[idx];
          ta.selectionStart = seg.startChar;
          ta.selectionEnd = seg.startChar + seg.raw.length;
        }
      });

      // Clicking in the raw SRT textarea can also seek if synced
      document.getElementById('srt-area').addEventListener('click', (e) => {
        if (!document.getElementById('sync-subs-to-video').checked) return;
        const ta = document.getElementById('srt-area');
        const pos = ta.selectionStart || 0;
        let idx = -1;
        for (let i=0;i<parsedSegments.length;i++) {
          const seg = parsedSegments[i];
          if (pos >= seg.startChar && pos <= seg.startChar + seg.raw.length) { idx = i; break; }
        }
        if (idx >= 0) {
          const seg = parsedSegments[idx];
          const v = document.getElementById('video');
          v.currentTime = seg.start + 0.01;
        }
      });
      async function updateStatus() {
        const res = await fetch('/status');
        const st = await res.json();
        const s = st[currentIndex] || {state:'idle'};
        document.getElementById('status').innerText = s.state + (s.msg ? ': ' + s.msg : '');
      }
      document.getElementById('done').addEventListener('click', async () => {
        const res = await fetch('/done', { method: 'POST' });
        const j = await res.json();
        if (j && j.ok) {
          // Try to close the window if possible; otherwise prompt the user.
          try { window.close(); } catch (err) { /* ignore */ }
          alert('GUI shutdown requested. Please close the tab if it is still open.');
        } else {
          alert('Failed to request GUI shutdown.');
        }
      });
      loadList().then(() => selectIndex(0));
      // Apply initial toggle state for segment editor
      document.getElementById('seg-edit-toggle').dispatchEvent(new Event('change'));
      setInterval(updateStatus, 2000);
      });
    </script>
  </body>
</html>
"""
