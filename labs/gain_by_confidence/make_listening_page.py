# ruff: noqa: RUF001  (the page text is Persian on purpose; ruff flags Persian letters and digits as look-alikes)
"""Builds index.html next to the numbered files written by recovery_suite.py, so they can be listened to side by side.

python make_listening_page.py out_dir
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

SCENARIOS = {
    "A": (
        "همه‌چیز افت می‌کند",
        "گفتار و نویز alex با هم افت می‌کنند، مثل کم‌شدن گین میکروفون.",
    ),
    "B": ("فقط گفتار افت می‌کند", "نویز alex سر جایش می‌ماند؛ در افت‌ها فقط نویز می‌ماند."),
    "C": (
        "گفتار افت‌کرده و نویز صورتی",
        "نویز صورتی با SNR برابر ۱۰ دسی‌بل نسبت به مرجع، بدون افت.",
    ),
    "D": (
        "گفتار افت‌کرده، هام و نویز سفید",
        "هام ۵۰ هرتز و نویز سفید با SNR برابر ۱۰ دسی‌بل نسبت به مرجع، بدون افت.",
    ),
}
# (row title, what it does, file stem for the normalization-only column, file stem for the denoise + normalization column)
METHODS = [
    (
        "RMS هم‌پوشان",
        "گین از سطح RMS پنجره‌ی ۱ ثانیه‌ای، با گام ۱۰ میلی‌ثانیه. غیر علّی.",
        "gain_rms_overlap",
        "gain_rms_overlap_then_gtcrn",
    ),
    (
        "f(p) × g(1/rms)",
        "همان گین، به‌اضافه‌ی کم‌شدن گین وقتی احتمال گفتار پایین است. غیر علّی.",
        "gain_fp_levelled",
        "gain_fp_levelled_then_gtcrn",
    ),
    (
        "استریم، f_min برابر ۱",
        "فریم‌به‌فریم و علّی؛ فقط AGC با هموارسازی h.",
        "gain_stream_fmin1",
        "gain_stream_fmin1_then_gtcrn",
    ),
    (
        "استریم، f_min برابر ۰٫۵",
        "علّی؛ در بی‌گفتاری گین تا نصف کم می‌شود.",
        "gain_stream_fmin0.5",
        "gain_stream_fmin0.5_then_gtcrn",
    ),
]
PAGE = """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>خروجی‌های آزمایش افت و بازیابی گین</title>
<style>
:root { color-scheme: light dark; --bg:#f4f6f7; --panel:#ffffff; --ink:#15212b; --muted:#566673; --line:#d3dce2;
  --signal:#0b6e8a; --signal-soft:#d9eef4; --best:#8a5a00; --best-soft:#fbefd2; }
@media (prefers-color-scheme: dark) { :root { --bg:#12181d; --panel:#1a232a; --ink:#e6edf1; --muted:#9db0bc;
  --line:#2c3a44; --signal:#5cc0dc; --signal-soft:#17323c; --best:#f0c25e; --best-soft:#3a2f10; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:17px/1.85 Vazirmatn, "Noto Naskh Arabic", Tahoma, system-ui, sans-serif; }
main { max-width:64rem; margin:0 auto; padding:2.5rem 1.25rem 4rem; }
h1 { font-size:1.9rem; line-height:1.4; margin:0 0 .75rem; }
h2 { font-size:1.35rem; margin:3rem 0 .25rem; }
h3 { font-size:1.05rem; margin:0; }
p { margin:.4rem 0; max-width:46rem; }
.muted { color:var(--muted); }
.how { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:1rem 1.25rem; margin:1.5rem 0; }
.how ol { margin:.4rem 0 0; padding-inline-start:1.4rem; }
.pair { display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem; margin-top:1rem; }
.pair > div, .cell { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:.75rem .9rem; }
audio { width:100%; margin-top:.4rem; }
.grid { display:grid; grid-template-columns:minmax(10rem, 1fr) 2fr 2fr 2fr; gap:.75rem; margin-top:1rem; align-items:stretch; }
.t { grid-column:1; grid-row:var(--r); } .g { grid-column:2; grid-row:var(--r); } .d { grid-column:3; } .b { grid-column:4; grid-row:var(--r); }
.head { font-weight:600; color:var(--muted); font-size:.95rem; padding:0 .25rem; }
.row-title { padding:.75rem .25rem; }
.row-title .muted { font-size:.9rem; line-height:1.6; }
.score { display:flex; align-items:center; gap:.6rem; font-variant-numeric:tabular-nums; }
.score b { min-width:3.6rem; }
.bar { flex:1; height:.5rem; background:var(--line); border-radius:3px; overflow:hidden; direction:ltr; }
.bar i { display:block; height:100%; background:var(--signal); }
.best { border-color:var(--best); background:var(--best-soft); }
.best .tag { color:var(--best); font-weight:600; font-size:.9rem; }
.cell .speed { font-size:.85rem; text-align:left; margin-top:.25rem; }
.cell .name { font-size:.85rem; color:var(--muted); direction:ltr; text-align:left; overflow-wrap:anywhere; }
a { color:var(--signal); }
:focus-visible { outline:3px solid var(--signal); outline-offset:2px; }
@media (max-width: 52rem) { .pair { grid-template-columns:1fr; } .grid { grid-template-columns:1fr; } .head { display:none; }
  .t, .g, .d, .b { grid-column:auto; grid-row:auto; }
  .cell::before { content:attr(data-col); display:block; font-weight:600; color:var(--muted); font-size:.9rem; } }
</style>
</head>
<body>
<main>
<h1>خروجی‌های آزمایش افت و بازیابی گین</h1>
<p>سیگنال alex را دینویز کردیم و خروجی را «مرجع تمیز» گرفتیم. بعد گفتار مرجع را جاهایی ضعیف کردیم، نویز اضافه کردیم و روش‌های گین را امتحان کردیم. اینجا می‌توانی همه را بشنوی و با مرجع مقایسه کنی.</p>

<div class="how">
<strong>چطور گوش بدهم</strong>
<ol>
<li>اول مرجع تمیز را بشنو تا بدانی هدف چه صدایی است.</li>
<li>یک سناریو را انتخاب کن و «بدون پردازش» را بشنو. این همان ورودی خراب است.</li>
<li>بعد سه ستون را بشنو: «فقط نرمال‌سازی»، «فقط دینویز» و «دینویز و نرمال‌سازی». ستون وسط به روش بستگی ندارد و برای همه‌ی ردیف‌ها یک فایل است.</li>
</ol>
<p><strong>عدد کنار هر فایل</strong> SI-SDR نسبت به مرجع است، بر حسب دسی‌بل. هرچه بیشتر باشد، خروجی به مرجع نزدیک‌تر است؛ یعنی هم سطح صدا درست‌تر شده و هم نویز کمتر مانده. بالاترین عدد هر سناریو علامت دارد.</p>
<p><strong>RTF</strong> زمان پردازش تقسیم بر طول صدا است؛ کمتر از ۱ یعنی سریع‌تر از زمان واقعی. زمان‌ها میانه‌ی ۳ اجرا روی یک هسته‌ی سندباکس‌اند و نسخه‌ی «استریم» یک شبیه‌سازی پایتونی است. جزئیات و همه‌ی عددها در <a href="results.json">results.json</a> است.</p>
<p class="muted">مرجع خودش خروجی یک دنوایزر (DNS64) است، نه ضبط استودیویی. پس عددها فقط داخل یک سناریو قابل مقایسه‌اند و صدای مرجع هم کامل نیست. فایل‌ها با پیشوند شماره (مثل A_06) نام‌گذاری شده‌اند.</p>
</div>

<h2>فایل‌های مشترک</h2>
<div class="pair">
__COMMON__
</div>

__SCENARIOS__
</main>
<script>
document.addEventListener("play", function (e) {
  document.querySelectorAll("audio").forEach(function (a) { if (a !== e.target) { a.pause(); } });
}, true);
</script>
</body>
</html>
"""


def scores_from_index(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\| (\S+\.wav) \| SI-SDR vs reference (-?[\d.]+) dB \|", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def speed_line(rec: dict | None) -> str:
    if not rec:
        return ""
    if rec["total_seconds"] == 0:
        return '<div class="speed muted">بدون پردازش</div>'
    return (
        f'<div class="speed muted" dir="ltr">RTF {rec["rtf_total"]:.4f}, {rec["total_seconds"]:.2f} s'
        f" (gain {rec['rtf_gain']:.4f}, denoise {rec['rtf_denoise']:.4f})</div>"
    )


def cell(
    col: str,
    fname: str,
    score: float,
    best: bool,
    rec: dict | None = None,
    cls: str = "",
    style: str = "",
) -> str:
    pct = min(max((score + 4) / 18 * 100, 2), 100)
    tag = '<div class="tag">بالاترین در این سناریو</div>' if best else ""
    return (
        f'<div class="cell{" best" if best else ""} {cls}" style="{style}" data-col="{html.escape(col)}">{tag}'
        f'<div class="score"><b>{score:.1f} dB</b><span class="bar"><i style="width:{pct:.0f}%"></i></span></div>'
        f'<audio controls preload="none" src="{html.escape(fname)}"></audio>'
        f'{speed_line(rec)}<div class="name">{html.escape(fname)}</div></div>'
    )


def main() -> None:
    out = Path(sys.argv[1])
    scores = scores_from_index(out / "INDEX.md")
    files = {p.name: p for p in out.glob("*.wav")}
    results = (
        json.loads((out / "results.json").read_text(encoding="utf-8"))
        if (out / "results.json").exists()
        else {}
    )
    recs = {
        r["file"]: r
        for sc in results.get("scenarios", {}).values()
        for r in sc["outputs"]
    }

    def find(key: str, stem: str) -> str:
        return next(
            n for n in files if n.startswith(f"{key}_") and n.endswith(f"_{stem}.wav")
        )

    common = ""
    for fname, title, text in (
        (
            "00_reference_clean.wav",
            "مرجع تمیز",
            "alex بعد از دینویز؛ هدف همه‌ی سناریوها.",
        ),
        (
            "01_noise_alex_extracted.wav",
            "نویز استخراج‌شده",
            "ورودی منهای مرجع؛ نویزی که در سناریوهای A و B استفاده شد.",
        ),
        ("02_alex_noisy_original.wav", "alex نویزی اصلی", "فایل اصلی، بدون افت."),
    ):
        common += (
            f'<div><h3>{title}</h3><p class="muted">{text}</p><audio controls preload="none" src="{fname}"></audio>'
            f'<div class="name muted" dir="ltr">{fname}</div></div>\n'
        )

    sections = ""
    for key, (title, text) in SCENARIOS.items():
        pairs = [(find(key, a), find(key, b)) for _, _, a, b in METHODS]
        mix_f, den_f = find(key, "mix_input"), find(key, "gtcrn_only")
        best_name = max(
            (mix_f, den_f, *(n for pair in pairs for n in pair)),
            key=lambda n: scores[n],
        )
        n_rows = len(METHODS)
        rows = (
            '<div class="head t" style="--r:1">روش</div><div class="head g" style="--r:1">فقط نرمال‌سازی</div>'
            '<div class="head d" style="grid-row:1">فقط دینویز</div><div class="head b" style="--r:1">دینویز و نرمال‌سازی</div>\n'
        )
        for i, ((mtitle, mtext, _, _), (fa, fb)) in enumerate(
            zip(METHODS, pairs, strict=True)
        ):
            r = i + 2
            rows += (
                f'<div class="row-title t" style="--r:{r}"><h3>{mtitle}</h3><div class="muted">{mtext}</div></div>\n'
                f"{cell('فقط نرمال‌سازی', fa, scores[fa], fa == best_name, recs.get(fa), 'g', f'--r:{r}')}\n"
                f"{cell('دینویز و نرمال‌سازی', fb, scores[fb], fb == best_name, recs.get(fb), 'b', f'--r:{r}')}\n"
            )
        rows += (
            cell(
                "فقط دینویز",
                den_f,
                scores[den_f],
                den_f == best_name,
                recs.get(den_f),
                "d",
                f"grid-row:2 / span {n_rows}",
            )
            + "\n"
        )
        inp = cell("ورودی", mix_f, scores[mix_f], mix_f == best_name, recs.get(mix_f))
        sections += (
            f'<h2>سناریوی {key}: {title}</h2>\n<p class="muted">{text}</p>\n'
            f'<h3 style="margin-top:1rem">ورودی خراب، بدون هیچ پردازشی</h3>\n<div style="max-width:30rem">{inp}</div>\n'
            f'<div class="grid">\n{rows}</div>\n'
        )

    (out / "index.html").write_text(
        PAGE.replace("__COMMON__", common).replace("__SCENARIOS__", sections),
        encoding="utf-8",
    )
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
