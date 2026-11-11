#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the result files.

    python3 readme_figure.py

Reads results/<model>/<processor>.json for the 52 measured files (the files named in pc/models.txt, without the
*_fixed copies). Writes docs/images/results-light.svg and docs/images/results-dark.svg, and prints the sentence the
figure's alt text uses.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "docs" / "images"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "pass": "#2a78d6", "other": "#eb6834"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "pass": "#3987e5", "other": "#d95926"},
}
TARGETS = [("ovnpu", "Intel NPU, compiled at load"), ("ovnpu_pre", "Intel NPU, precompiled"), ("ovgpu", "Intel GPU"),
           ("cuda16", "RTX, fp16"), ("cuda32", "RTX, fp32")]
KEYS = [("pass", "Output stayed close to the CPU's"), ("other", "Output differed, or the file did not run")]
TITLE = "Results for 52 files on each processor"
WIDTH, BAR, GAP = 860, 24, 2


def count():
    """Return {cell: {pass, fail, error}} over the 52 measured files."""
    source = ROOT / "pc" / "models.txt"
    if not source.is_file():
        sys.exit(f"readme_figure: {source} not found")
    models = [m for m in source.read_text(encoding="utf-8").split() if not m.endswith("_fixed")]
    if len(models) != 52:
        sys.exit(f"readme_figure: expected 52 files in {source}, found {len(models)}")
    counts = {}
    for cell, _ in TARGETS:
        c = {"pass": 0, "fail": 0, "error": 0}
        for m in models:
            path = ROOT / "results" / m / f"{cell}.json"
            if not path.is_file():
                sys.exit(f"readme_figure: {path} not found")
            status = json.loads(path.read_text(encoding="utf-8"))["status"]
            if status not in ("PASS", "FAIL", "ERROR"):
                sys.exit(f"readme_figure: unknown status {status} in {path}")
            c[status.lower()] += 1
        counts[cell] = c
    return counts


def text(x, y, body, size, fill, weight="400"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}">{body}</text>'


def summary(v):
    return f"{v['pass']} passed, {v['fail']} differed, {v['error']} did not run"


def draw(counts, theme, desc):
    c = THEMES[theme]
    unit = WIDTH / 52
    parts = [text(0, 28, TITLE, 20, c["text"], "600")]
    y = 56
    for key, label in KEYS:
        parts.append(f'<rect x="0" y="{y - 12:g}" width="14" height="14" rx="4" fill="{c[key]}"/>')
        parts.append(text(20, y, label, 14, c["muted"]))
        y += 22
    y += 18
    for cell, label in TARGETS:
        v = counts[cell]
        parts.append(text(0, y, f"{label}: {summary(v)}", 15, c["text"]))
        x = 0
        for key, n in (("pass", v["pass"]), ("other", v["fail"] + v["error"])):
            if n:
                parts.append(f'<rect x="{x:g}" y="{y + 8:g}" width="{n * unit - GAP:g}" height="{BAR}" rx="4" fill="{c[key]}"/>')
                x += n * unit
        y += 62
    height = y - 62 + 8 + BAR + 8
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    counts = count()
    desc = " ".join(f"{label}: {summary(counts[c])}." for c, label in TARGETS)
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"results-{theme}.svg").write_text(draw(counts, theme, desc), encoding="utf-8")
    print(desc)


if __name__ == "__main__":
    main()
