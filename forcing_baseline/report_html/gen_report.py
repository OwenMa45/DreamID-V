"""Build a self-contained HTML report for the causal face-swap results.

For each result video under ``outputs/stage<N>/`` this reconstructs the original
data group (source video + reference image) from the filename, copies all three
assets into ``<report_dir>/assets/`` and emits a single ``index.html`` that shows
them side by side (source video | reference face | swapped result) for review.

Result filenames are produced by ``scripts/infer_latest_2h100.sh`` as::

    <sname>_step<NNNNNN>_g<NN>_<base>.mp4
    e.g. stage1_ar_step005000_g00_part_004_0000000_..._id0.mp4

and the matching originals live in the Stage-0 input dir as::

    <base>.mp4        source / driving video
    <base>_ref.jpg    reference identity face

Because the whole report (html + assets) is self-contained, the directory can be
zipped or served with any static file server and opened on another machine.

Example
-------
    python report_html/gen_report.py \
        --outputs_dir outputs \
        --input_dir  /path/to/part_004/input \
        --report_dir report_html/report \
        --stages all
"""
import argparse
import os
import re
import shutil
from datetime import datetime

# stage1_ar_step005000_g00_<base>.mp4  ->  groups: sname, step, grp, base
RESULT_RE = re.compile(
    r"^(?P<sname>stage\d+_[a-zA-Z]+)_step(?P<step>\d+)_g(?P<grp>\d+)_(?P<base>.+)\.mp4$")

STAGE_TITLES = {
    "stage1": "Stage 1 · Autoregressive Diffusion (AR)",
    "stage2": "Stage 2 · Consistency Distillation (CD)",
    "stage3": "Stage 3 · DMD (final few-step)",
}


def _find_source(input_dir, base, ref_suffix, video_ext):
    """Return (video_path, ref_path) for a base name, or (None, None)."""
    video = os.path.join(input_dir, base + video_ext)
    ref = os.path.join(input_dir, base + ref_suffix)
    return (video if os.path.isfile(video) else None,
            ref if os.path.isfile(ref) else None)


def _copy_asset(src, assets_dir, dst_name):
    """Copy src into assets_dir under dst_name; return the basename (or None)."""
    if not src or not os.path.isfile(src):
        return None
    dst = os.path.join(assets_dir, dst_name)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    return dst_name


def collect(outputs_dir, input_dir, stages, assets_dir, ref_suffix, video_ext):
    """Scan the requested stage folders and build the per-stage row lists."""
    if stages in ("all", "", None):
        stage_dirs = sorted(
            d for d in os.listdir(outputs_dir)
            if d.startswith("stage") and os.path.isdir(os.path.join(outputs_dir, d)))
    else:
        stage_dirs = [f"stage{s.strip().lstrip('stage')}" for s in stages.split(",")]

    report = []  # list of (stage_key, title, [rows])
    for stage in stage_dirs:
        sdir = os.path.join(outputs_dir, stage)
        if not os.path.isdir(sdir):
            print(f"[report] skip missing stage dir: {sdir}")
            continue
        rows = []
        for fname in sorted(os.listdir(sdir)):
            m = RESULT_RE.match(fname)
            if not m:
                continue
            base = m.group("base")
            step, grp = m.group("step"), m.group("grp")
            video_src, ref_src = _find_source(input_dir, base, ref_suffix, video_ext)

            prefix = f"{stage}_g{grp}"
            res_rel = _copy_asset(os.path.join(sdir, fname), assets_dir, f"{prefix}_result.mp4")
            vid_rel = _copy_asset(video_src, assets_dir, f"{prefix}_source.mp4")
            ref_rel = _copy_asset(ref_src, assets_dir, f"{prefix}_ref.jpg")

            rows.append({
                "group": grp, "step": step, "base": base,
                "result": res_rel, "source": vid_rel, "ref": ref_rel,
                "missing": [n for n, v in
                            (("source video", vid_rel), ("reference image", ref_rel)) if v is None],
            })
        if rows:
            report.append((stage, STAGE_TITLES.get(stage, stage), rows))
        else:
            print(f"[report] no matching result mp4 in {sdir}")
    return report


def _cell_video(rel, label):
    if rel:
        return (f'<div class="cell"><span class="lab">{label}</span>'
                f'<video src="assets/{rel}" controls loop muted playsinline></video></div>')
    return f'<div class="cell missing"><span class="lab">{label}</span><div class="ph">missing</div></div>'


def _cell_image(rel, label):
    if rel:
        return (f'<div class="cell"><span class="lab">{label}</span>'
                f'<img src="assets/{rel}" alt="{label}"></div>')
    return f'<div class="cell missing"><span class="lab">{label}</span><div class="ph">missing</div></div>'


def render_html(report, title):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = []
    total = 0
    for stage, stitle, rows in report:
        cards = []
        for r in rows:
            total += 1
            note = ""
            if r["missing"]:
                note = f'<div class="note">missing: {", ".join(r["missing"])}</div>'
            cards.append(f"""
        <div class="card">
          <div class="card-head">
            <span class="grp">group {int(r['group'])}</span>
            <span class="step">step {int(r['step'])}</span>
            <span class="base" title="{r['base']}">{r['base']}</span>
          </div>
          <div class="row">
            {_cell_video(r['source'], 'Source video')}
            {_cell_image(r['ref'], 'Reference face')}
            {_cell_video(r['result'], 'Swapped result')}
          </div>
          {note}
        </div>""")
        sections.append(
            f'<section><h2>{stitle}</h2><div class="cards">{"".join(cards)}</div></section>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg:#0f1115; --card:#171a21; --line:#262b36; --fg:#e8eaed; --mut:#9aa4b2; --acc:#4f9dff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:24px 32px; border-bottom:1px solid var(--line); position:sticky; top:0;
           background:rgba(15,17,21,.92); backdrop-filter:blur(6px); z-index:10; }}
  header h1 {{ margin:0 0 4px; font-size:20px; }}
  header .meta {{ color:var(--mut); font-size:13px; }}
  main {{ padding:24px 32px 64px; max-width:1400px; margin:0 auto; }}
  section {{ margin-bottom:40px; }}
  section h2 {{ font-size:16px; color:var(--acc); border-left:3px solid var(--acc);
               padding-left:10px; margin:0 0 16px; }}
  .cards {{ display:flex; flex-direction:column; gap:18px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }}
  .card-head {{ display:flex; align-items:center; gap:12px; margin-bottom:12px; flex-wrap:wrap; }}
  .card-head .grp {{ font-weight:600; }}
  .card-head .step {{ color:var(--mut); font-size:12px; background:#20252f; padding:2px 8px; border-radius:20px; }}
  .card-head .base {{ color:var(--mut); font-size:12px; font-family:ui-monospace,Menlo,Consolas,monospace;
                     overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:60%; }}
  .row {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
  .cell {{ display:flex; flex-direction:column; gap:6px; }}
  .cell .lab {{ font-size:12px; color:var(--mut); }}
  .cell video, .cell img {{ width:100%; border-radius:8px; background:#000; aspect-ratio:1/1; object-fit:contain; }}
  .cell.missing .ph {{ display:flex; align-items:center; justify-content:center; aspect-ratio:1/1;
                      border:1px dashed var(--line); border-radius:8px; color:var(--mut); font-size:13px; }}
  .note {{ margin-top:8px; color:#f0b429; font-size:12px; }}
  @media (max-width:820px) {{ .row {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">Generated {ts} · {total} group(s) · source video ｜ reference face ｜ swapped result</div>
</header>
<main>
{"".join(sections) if sections else '<p style="color:#9aa4b2">No results found. Run scripts/infer_latest_2h100.sh first.</p>'}
</main>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser(description="Generate a self-contained HTML face-swap report")
    p.add_argument("--outputs_dir", default="outputs", help="Dir containing stage<N>/ result folders")
    p.add_argument("--input_dir", required=True, help="Stage-0 input dir with <base>{.mp4,_ref.jpg}")
    p.add_argument("--report_dir", default="report_html/report", help="Output report directory")
    p.add_argument("--stages", default="all", help="'all' or comma list e.g. '1,3' / 'stage1,stage3'")
    p.add_argument("--title", default="Causal DreamID-V · Face-Swap Report")
    p.add_argument("--ref_suffix", default="_ref.jpg")
    p.add_argument("--video_ext", default=".mp4")
    args = p.parse_args()

    assets_dir = os.path.join(args.report_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    report = collect(args.outputs_dir, args.input_dir, args.stages,
                     assets_dir, args.ref_suffix, args.video_ext)
    html = render_html(report, args.title)

    index = os.path.join(args.report_dir, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(html)

    n_groups = sum(len(rows) for _, _, rows in report)
    print(f"[report] {len(report)} stage(s), {n_groups} group(s) -> {index}")
    print(f"[report] assets copied into {assets_dir}")
    print(f"[report] open it with:  python -m http.server -d {args.report_dir} 8000")


if __name__ == "__main__":
    main()
