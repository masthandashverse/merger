from flask import Flask, render_template, request, send_file, jsonify, Response
import subprocess
import os
import uuid
import shutil
import re
import tempfile
import json
import threading
import time
from pathlib import Path

app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024

os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()


# ===========================================================
# SYSTEM CHECKS
# ===========================================================

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def check_filters():
    try:
        r = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True, timeout=10)
        out = r.stdout
        return {
            'subtitles': any('subtitles' in l and 'V->V' in l for l in out.split('\n')),
            'ass':       any(' ass '    in l and 'V->V' in l for l in out.split('\n')),
        }
    except Exception:
        return {'subtitles': False, 'ass': False}


# ===========================================================
# SRT UTILITIES
# ===========================================================

def parse_srt(path):
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    content = None
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except Exception:
            continue
    if not content:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

    content = content.replace('\ufeff', '').replace('\r\n', '\n').replace('\r', '\n')
    entries = []

    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        time_match = None
        time_idx   = -1
        for li, line in enumerate(lines):
            m = re.match(
                r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*'
                r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})', line.strip()
            )
            if m:
                time_match = m
                time_idx   = li
                break
        if not time_match:
            continue
        g     = time_match.groups()
        start = int(g[0])*3600 + int(g[1])*60 + int(g[2]) + int(g[3])/1000
        end   = int(g[4])*3600 + int(g[5])*60 + int(g[6]) + int(g[7])/1000
        text  = '\n'.join(lines[time_idx+1:])
        text  = re.sub(r'<[^>]+>', '', text)
        text  = re.sub(r'\{[^}]+\}', '', text).strip()
        if text:
            entries.append({'start': start, 'end': end, 'text': text})
    return entries


def detect_language(text):
    """Returns 'cjk' if text is mostly Chinese/Japanese/Korean, else 'latin'."""
    cjk_count = 0
    total     = 0
    for ch in text:
        cp     = ord(ch)
        total += 1
        if (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or
            0xF900 <= cp <= 0xFAFF or
            0x2E80 <= cp <= 0x2EFF or
            0x3000 <= cp <= 0x303F or
            0xFF00 <= cp <= 0xFFEF or
            0xAC00 <= cp <= 0xD7AF or
            0x3040 <= cp <= 0x309F or
            0x30A0 <= cp <= 0x30FF):
            cjk_count += 1
    if total == 0:
        return 'latin'
    return 'cjk' if (cjk_count / total) > 0.2 else 'latin'


def split_cjk_and_latin(text):
    """Split subtitle block into (cjk_string, latin_string)."""
    lines      = [l.strip() for l in text.split('\n') if l.strip()]
    cjk_lines  = []
    lat_lines  = []
    for line in lines:
        if detect_language(line) == 'cjk':
            cjk_lines.append(line)
        else:
            lat_lines.append(line)
    return '\n'.join(cjk_lines), '\n'.join(lat_lines)


def format_ass_time(s):
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    cs = int((s % 1) * 100)
    return f"{h}:{m:02d}:{sc:02d}.{cs:02d}"


def format_srt_time(s):
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sc:02d},{ms:03d}"


def create_ass(srt_path, ass_path, w=1920, h=1080):
    """
    Build a dual-style ASS file.

    Screen layout (bottom → top):
        [ Chinese subtitle ]   ← chinese_margin_v from bottom
        [ English subtitle ]   ← english_margin_v from bottom  (above Chinese)
    """
    entries = parse_srt(srt_path)

    chinese_font_size = max(int(h * 0.052), 28)
    english_font_size = max(int(h * 0.038), 20)
    margin_lr         = int(w * 0.06)

    # Chinese sits above the bottom watermark area
    chinese_margin_v  = int(h * 0.18)          # ← tweak to move up/down

    # English sits ABOVE Chinese
    gap              = int(h * 0.015)
    english_margin_v = chinese_margin_v + chinese_font_size + gap

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\n"
        f"PlayResY: {h}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"

        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"

        # Chinese — bold white, black outline+shadow, bottom-center
        f"Style: Chinese,"
        f"Arial Unicode MS,{chinese_font_size},"
        f"&H00FFFFFF,&H000000FF,&H00000000,&H96000000,"
        f"-1,0,0,0,"
        f"100,100,0,0,"
        f"1,3,1,"
        f"2,"
        f"{margin_lr},{margin_lr},{chinese_margin_v},1\n"

        # English — normal weight, slightly smaller, above Chinese
        f"Style: English,"
        f"Arial,{english_font_size},"
        f"&H00FFFFFF,&H000000FF,&H00000000,&H96000000,"
        f"0,0,0,0,"
        f"100,100,0,0,"
        f"1,2,1,"
        f"2,"
        f"{margin_lr},{margin_lr},{english_margin_v},1\n\n"

        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )

    dialogue_lines = []

    for e in entries:
        start    = format_ass_time(e['start'])
        end      = format_ass_time(e['end'])
        raw      = e['text']
        cjk, lat = split_cjk_and_latin(raw)

        if cjk and lat:
            dialogue_lines.append(
                f"Dialogue: 0,{start},{end},Chinese,,0,0,0,,{cjk.replace(chr(10), chr(92)+'N')}\n"
            )
            dialogue_lines.append(
                f"Dialogue: 0,{start},{end},English,,0,0,0,,{lat.replace(chr(10), chr(92)+'N')}\n"
            )
        elif cjk:
            dialogue_lines.append(
                f"Dialogue: 0,{start},{end},Chinese,,0,0,0,,{cjk.replace(chr(10), chr(92)+'N')}\n"
            )
        else:
            fallback = (lat or raw).replace('\n', '\\N')
            dialogue_lines.append(
                f"Dialogue: 0,{start},{end},English,,0,0,0,,{fallback}\n"
            )

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(dialogue_lines)

    return len(entries)


def clean_srt(src, dst):
    entries = parse_srt(src)
    with open(dst, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries, 1):
            f.write(f"{i}\n{format_srt_time(e['start'])} --> "
                    f"{format_srt_time(e['end'])}\n{e['text']}\n\n")
    return len(entries)


def get_video_info(path):
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', path],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            w, h, dur = 1920, 1080, 0
            for s in data.get('streams', []):
                if s.get('codec_type') == 'video':
                    w   = int(s.get('width',  1920))
                    h   = int(s.get('height', 1080))
                    dur = float(s.get('duration', 0))
            if dur == 0:
                dur = float(data.get('format', {}).get('duration', 0))
            return {'width': w, 'height': h, 'duration': dur}
    except Exception as ex:
        print(f"[get_video_info] {ex}")
    return {'width': 1920, 'height': 1080, 'duration': 0}


# ===========================================================
# PROGRESS HELPERS
# ===========================================================

def set_progress(batch_id, ep_idx, pct, msg, status='processing'):
    with jobs_lock:
        if batch_id not in jobs:
            jobs[batch_id] = {}
        jobs[batch_id][f'ep_{ep_idx}'] = {
            'pct': pct, 'msg': msg, 'status': status, 'ts': time.time()
        }


def set_done(batch_id, results, dl_folder):
    with jobs_lock:
        jobs[batch_id]['_final'] = {
            'results':         results,
            'download_folder': dl_folder,
            'ts':              time.time()
        }


# ===========================================================
# FFMPEG HELPERS
# ===========================================================

def run_ff(cmd, cwd=None, timeout=7200):
    print(f"[FFmpeg] {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
        if result.returncode != 0:
            print(f"[FFmpeg] FAILED:\n{result.stderr[-400:]}")
        else:
            print("[FFmpeg] OK")
        return result
    except subprocess.TimeoutExpired:
        print("[FFmpeg] TIMEOUT")
        return None
    except Exception as ex:
        print(f"[FFmpeg] EXCEPTION: {ex}")
        return None


def good(result, out_path):
    if not result or result.returncode != 0:
        return False
    if not os.path.exists(out_path):
        return False
    size = os.path.getsize(out_path)
    if size < 10000:
        print(f"[good] output too small: {size} bytes")
        return False
    return True


# ===========================================================
# HARD-SUB METHODS
# ===========================================================

def method_ass(video, sub, out, sub_ext, work_dir, info):
    """Primary: SRT → dual-style ASS → burn with ass= filter."""
    ass = os.path.join(work_dir, 'styled.ass')
    n   = create_ass(os.path.abspath(sub), ass, info['width'], info['height'])
    print(f"[method_ass] {n} entries → {ass}")
    if n == 0:
        return None

    ass_abs = os.path.abspath(ass)
    ass_esc = ass_abs.replace('\\', '/').replace(':', '\\:')

    return run_ff([
        'ffmpeg', '-y',
        '-i', os.path.abspath(video),
        '-vf', f"ass='{ass_esc}'",
        '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        os.path.abspath(out)
    ])


def method_subtitles_filter(video, sub, out, sub_ext, work_dir, info):
    """Fallback: subtitles= / ass= filter with absolute path."""
    abs_sub = os.path.abspath(sub)
    sub_esc = abs_sub.replace('\\', '/').replace(':', '\\:')
    vf = (f"ass='{sub_esc}'"
          if sub_ext in ('.ass', '.ssa')
          else f"subtitles='{sub_esc}'")
    return run_ff([
        'ffmpeg', '-y',
        '-i', os.path.abspath(video),
        '-vf', vf,
        '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        os.path.abspath(out)
    ])


def method_ffmpeg_ass_convert(video, sub, out, sub_ext, work_dir, info):
    """Fallback: ffmpeg converts SRT→ASS internally, then burn."""
    conv = os.path.join(work_dir, 'conv.ass')
    r    = run_ff(['ffmpeg', '-y', '-i', os.path.abspath(sub), conv], timeout=60)
    if not (r and r.returncode == 0 and os.path.exists(conv)):
        return None
    esc = os.path.abspath(conv).replace('\\', '/').replace(':', '\\:')
    return run_ff([
        'ffmpeg', '-y',
        '-i', os.path.abspath(video),
        '-vf', f"ass='{esc}'",
        '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        os.path.abspath(out)
    ])


def method_mux_burn(video, sub, out, sub_ext, work_dir, info):
    """Fallback: mux into temp MKV then burn subtitle from it."""
    tmp = os.path.join(work_dir, f'_tmp_{uuid.uuid4().hex[:6]}.mkv')
    r1  = run_ff([
        'ffmpeg', '-y',
        '-i', os.path.abspath(video),
        '-i', os.path.abspath(sub),
        '-c', 'copy', '-c:s', 'srt',
        '-map', '0:v', '-map', '0:a?', '-map', '1:0', tmp
    ], timeout=600)
    if not (r1 and r1.returncode == 0):
        return r1
    esc    = tmp.replace('\\', '/').replace(':', '\\:')
    result = run_ff([
        'ffmpeg', '-y', '-i', tmp,
        '-vf', f"subtitles='{esc}'",
        '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        os.path.abspath(out)
    ])
    try:
        os.remove(tmp)
    except Exception:
        pass
    return result


# ===========================================================
# SOFT-SUB METHODS
# ===========================================================

def method_soft_mkv(video, sub, out, sub_ext):
    """Embed subtitle as selectable track in MKV."""
    mkv       = str(Path(out).with_suffix('.mkv'))
    sub_codec = 'ass' if sub_ext in ('.ass', '.ssa') else 'srt'
    r = run_ff([
        'ffmpeg', '-y',
        '-i', os.path.abspath(video),
        '-i', os.path.abspath(sub),
        '-map', '0:v', '-map', '0:a?', '-map', '1:0',
        '-c:v', 'copy', '-c:a', 'copy', '-c:s', sub_codec,
        '-metadata:s:s:0', 'language=eng',
        '-disposition:s:0', 'default',
        mkv
    ], timeout=3600)
    return r, mkv


def method_soft_mp4(video, sub, out, sub_ext, work_dir):
    """Embed subtitle as mov_text in MP4."""
    clean = os.path.join(work_dir, 'clean.srt')
    clean_srt(os.path.abspath(sub), clean)
    r = run_ff([
        'ffmpeg', '-y',
        '-i', os.path.abspath(video),
        '-i', clean,
        '-map', '0:v', '-map', '0:a?', '-map', '1:0',
        '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text',
        '-metadata:s:s:0', 'language=eng',
        '-disposition:s:0', 'default',
        '-movflags', '+faststart',
        os.path.abspath(out)
    ], timeout=3600)
    return r


# ===========================================================
# CORE PROCESSOR
# ===========================================================

def process_episode(video_path, srt_path, ep_name, merge_type,
                    output_dir, batch_id, ep_idx, dl_folder):
    work_dir = None
    try:
        set_progress(batch_id, ep_idx, 2, 'Starting…')

        # ── Validate ──────────────────────────────────────
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.isfile(srt_path):
            raise FileNotFoundError(f"Subtitle not found: {srt_path}")
        if os.path.getsize(video_path) < 10000:
            raise ValueError("Video file too small")
        if os.path.getsize(srt_path) < 5:
            raise ValueError("Subtitle file too small")

        entries = parse_srt(srt_path)
        if not entries:
            raise ValueError("No subtitle entries found — check SRT format")

        cjk_count   = sum(1 for e in entries if detect_language(e['text']) == 'cjk')
        latin_count = len(entries) - cjk_count
        print(f"[EP {ep_idx}] {len(entries)} subs  "
              f"({cjk_count} CJK / {latin_count} Latin)")

        set_progress(batch_id, ep_idx, 8,
                     f'Parsed {len(entries)} subtitles '
                     f'({cjk_count} CJK / {latin_count} Latin)')

        info     = get_video_info(video_path)
        sub_ext  = Path(srt_path).suffix.lower()
        safe     = re.sub(r'[<>:"/\\|?*]', '_', ep_name)
        safe     = re.sub(r'_+', '_', safe).strip('_') or f'episode_{ep_idx+1}'
        work_dir = tempfile.mkdtemp(prefix=f'ep{ep_idx}_')
        filters  = check_filters()

        # ── HARD SUBTITLES ─────────────────────────────────
        if merge_type == 'hard':
            out_file = os.path.join(output_dir, f'{batch_id}_ep{ep_idx}.mp4')
            out_name = f'{safe}.mp4'

            methods = []
            if filters.get('ass') or filters.get('subtitles'):
                methods += [
                    ('Dual-style ASS burn',  method_ass),
                    ('Subtitle filter',       method_subtitles_filter),
                    ('FFmpeg ASS convert',    method_ffmpeg_ass_convert),
                ]
            if filters.get('subtitles'):
                methods.append(('MKV mux+burn', method_mux_burn))

            if not methods:
                raise RuntimeError(
                    "No FFmpeg subtitle filters available — "
                    "reinstall FFmpeg with libass support."
                )

            success  = False
            last_err = 'No methods ran'
            n        = len(methods)

            for mi, (mname, mfn) in enumerate(methods):
                pct = 10 + int(mi / n * 78)
                set_progress(batch_id, ep_idx, pct, f'[{mi+1}/{n}] {mname}…')
                print(f"[EP {ep_idx}] Trying: {mname}")
                try:
                    result = mfn(video_path, srt_path, out_file,
                                 sub_ext, work_dir, info)
                    if good(result, out_file):
                        print(f"[EP {ep_idx}] ✅ {mname} succeeded")
                        success = True
                        break
                    last_err = (result.stderr[-300:]
                                if result and result.stderr else 'no output')
                    if os.path.exists(out_file):
                        os.remove(out_file)
                except Exception as ex:
                    last_err = str(ex)
                    print(f"[EP {ep_idx}] ❌ {mname}: {ex}")

            if not success:
                raise RuntimeError(
                    f"All hard-sub methods failed.\nLast error: {last_err[:300]}"
                )

        # ── SOFT SUBTITLES ─────────────────────────────────
        else:
            out_file = None
            out_name = None
            success  = False
            last_err = ''

            set_progress(batch_id, ep_idx, 20, 'Soft sub → MKV…')
            try:
                base   = os.path.join(output_dir, f'{batch_id}_ep{ep_idx}.mp4')
                r, mkv = method_soft_mkv(video_path, srt_path, base, sub_ext)
                if good(r, mkv):
                    out_file = mkv
                    out_name = f'{safe}.mkv'
                    success  = True
                else:
                    last_err = r.stderr[-200:] if r and r.stderr else 'mkv failed'
            except Exception as ex:
                last_err = str(ex)

            if not success:
                set_progress(batch_id, ep_idx, 50, 'Soft sub → MP4…')
                try:
                    mp4 = os.path.join(output_dir, f'{batch_id}_ep{ep_idx}.mp4')
                    r   = method_soft_mp4(video_path, srt_path,
                                          mp4, sub_ext, work_dir)
                    if good(r, mp4):
                        out_file = mp4
                        out_name = f'{safe}.mp4'
                        success  = True
                    else:
                        last_err = r.stderr[-200:] if r and r.stderr else 'mp4 failed'
                except Exception as ex:
                    last_err = str(ex)

            if not success:
                raise RuntimeError(f"Soft-sub failed: {last_err[:300]}")

        # ── Save to dl_folder ──────────────────────────────
        set_progress(batch_id, ep_idx, 92, 'Saving…')
        dest    = os.path.join(dl_folder, out_name)
        shutil.copy2(out_file, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024

        set_progress(batch_id, ep_idx, 100,
                     f'✅ Done! {size_mb:.1f} MB → {out_name}', 'completed')

        return {
            'success':     True,
            'filename':    out_name,
            'path':        dest,
            'server_path': out_file,
            'size_mb':     round(size_mb, 1)
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        msg = str(exc)
        set_progress(batch_id, ep_idx, 100, f'❌ {msg[:180]}', 'error')
        return {'success': False, 'error': msg, 'filename': ep_name}

    finally:
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


# ===========================================================
# ROUTES
# ===========================================================

@app.route('/')
def index():
    return render_template('index.html',
                           ffmpeg_ok=check_ffmpeg(),
                           filters=check_filters())


@app.route('/debug')
def debug():
    info = {
        'ffmpeg':        check_ffmpeg(),
        'filters':       check_filters(),
        'output_dir':    os.path.abspath(app.config['OUTPUT_FOLDER']),
        'output_exists': os.path.isdir(app.config['OUTPUT_FOLDER']),
    }
    try:
        r = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        info['ffmpeg_version'] = r.stdout.split('\n')[0]
    except Exception as ex:
        info['ffmpeg_version'] = str(ex)
    return jsonify(info)


@app.route('/merge', methods=['POST'])
def merge():
    try:
        if not check_ffmpeg():
            return jsonify({'error': 'FFmpeg not installed or not in PATH'}), 500

        merge_type = request.form.get('merge_type', 'hard')
        ep_count   = int(request.form.get('episode_count', 0))

        print(f"\n[/merge] merge_type={merge_type}  ep_count={ep_count}")

        if ep_count == 0:
            return jsonify({'error': 'No episodes provided'}), 400

        batch_id   = uuid.uuid4().hex[:8]
        output_dir = app.config['OUTPUT_FOLDER']
        upload_dir = tempfile.mkdtemp(prefix=f'upload_{batch_id}_')

        # Files are served from output_dir via /download — no separate dl_folder
        dl_folder  = output_dir

        episodes = []
        for i in range(ep_count):
            vf = request.files.get(f'video_{i}')
            sf = request.files.get(f'srt_{i}')
            nm = request.form.get(f'ep_name_{i}', f'Episode_{i+1}').strip()

            if not vf or not vf.filename or not sf or not sf.filename:
                print(f"[/merge] ep {i}: missing file — skip")
                continue

            v_ext  = Path(vf.filename).suffix.lower() or '.mp4'
            s_ext  = Path(sf.filename).suffix.lower() or '.srt'
            v_path = os.path.join(upload_dir, f'video_{i}{v_ext}')
            s_path = os.path.join(upload_dir, f'srt_{i}{s_ext}')
            vf.save(v_path)
            sf.save(s_path)

            print(f"[/merge] ep {i}: {nm!r}  "
                  f"video={os.path.getsize(v_path):,}b  "
                  f"srt={os.path.getsize(s_path):,}b")

            episodes.append({
                'video': v_path,
                'srt':   s_path,
                'name':  nm or f'Episode_{i+1}'
            })

        if not episodes:
            shutil.rmtree(upload_dir, ignore_errors=True)
            return jsonify({'error': 'No valid episodes — each needs video + SRT'}), 400

        with jobs_lock:
            jobs[batch_id] = {
                f'ep_{i}': {
                    'pct':    0,
                    'msg':    'Queued…',
                    'status': 'queued',
                    'ts':     time.time()
                }
                for i in range(len(episodes))
            }

        def run_batch():
            results = []
            for i, ep in enumerate(episodes):
                r = process_episode(
                    ep['video'], ep['srt'], ep['name'],
                    merge_type, output_dir, batch_id, i, dl_folder
                )
                results.append(r)
            set_done(batch_id, results, dl_folder)
            shutil.rmtree(upload_dir, ignore_errors=True)

        threading.Thread(target=run_batch, daemon=True).start()

        return jsonify({
            'batch_id': batch_id,
            'ep_count': len(episodes),
        })

    except Exception as ex:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(ex)}), 500


@app.route('/progress/<batch_id>')
def progress(batch_id):
    with jobs_lock:
        return jsonify(jobs.get(batch_id, {}))


@app.route('/progress_stream/<batch_id>')
def progress_stream(batch_id):
    def gen():
        last = None
        t    = 0
        while t < 7200:
            with jobs_lock:
                data = dict(jobs.get(batch_id, {}))
            s = json.dumps(data)
            if s != last:
                yield f"data: {s}\n\n"
                last = s
                if '_final' in data:
                    break
            time.sleep(0.8)
            t += 0.8
        yield 'data: {"_done":true}\n\n'

    return Response(
        gen(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/download/<batch_id>/<int:ep_idx>')
def download(batch_id, ep_idx):
    """Serve the merged file directly to the browser."""
    for ext in ('.mp4', '.mkv'):
        p = os.path.join(app.config['OUTPUT_FOLDER'],
                         f'{batch_id}_ep{ep_idx}{ext}')
        if os.path.exists(p):
            with jobs_lock:
                fin     = jobs.get(batch_id, {}).get('_final', {})
                results = fin.get('results', [])
            fname = (
                results[ep_idx].get('filename', f'episode_{ep_idx+1}{ext}')
                if ep_idx < len(results) else f'episode_{ep_idx+1}{ext}'
            )
            mime = 'video/x-matroska' if ext == '.mkv' else 'video/mp4'
            return send_file(
                os.path.abspath(p),
                as_attachment=True,
                download_name=fname,
                mimetype=mime
            )
    return jsonify({'error': 'File not found — it may have been cleaned up'}), 404


@app.route('/cleanup', methods=['POST'])
def cleanup():
    now, removed = time.time(), 0
    for f in os.listdir(app.config['OUTPUT_FOLDER']):
        fp = os.path.join(app.config['OUTPUT_FOLDER'], f)
        if os.path.isfile(fp) and now - os.path.getmtime(fp) > 3600:
            try:
                os.remove(fp)
                removed += 1
            except Exception:
                pass
    return jsonify({'removed': removed})


# ===========================================================
# ERROR HANDLERS
# ===========================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 50 GB)'}), 413


@app.errorhandler(500)
def server_error(e):
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500


# ===========================================================
# MAIN
# ===========================================================

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    print("=" * 55)
    print("  🎬  VIDEO + SUBTITLE MERGER")
    print("=" * 55)
    ok = check_ffmpeg()
    print(f"  FFmpeg  : {'✅ found' if ok else '❌ NOT FOUND'}")
    if ok:
        for k, v in check_filters().items():
            print(f"  {k:12s}: {'✅' if v else '❌'}")
    print(f"  Port    : {port}")
    print("=" * 55)
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)