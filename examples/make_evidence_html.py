"""Render the captured demo/trace output into polished, email-ready HTML pages.

Produces four standalone HTML files under images/_html/ which are then
screenshotted to PNGs. Pure standard library; no third-party dependency.
"""
from __future__ import annotations

import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "images", "_html")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared styling — a clean "terminal card" look that reads well in an email.
# ---------------------------------------------------------------------------
CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
  background: #0d1117; color: #e6edf3; padding: 40px; width: 1180px;
}
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
  overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,.4); }
.titlebar { background: #21262d; padding: 14px 20px; display: flex;
  align-items: center; gap: 8px; border-bottom: 1px solid #30363d; }
.dot { width: 12px; height: 12px; border-radius: 50%; }
.r{background:#ff5f56}.y{background:#ffbd2e}.g{background:#27c93f}
.titlebar .name { margin-left: 12px; color: #8b949e; font-size: 14px;
  font-family: 'SF Mono', Menlo, monospace; }
h1 { font-size: 26px; padding: 26px 30px 4px; font-weight: 700; }
.sub { color: #8b949e; font-size: 15px; padding: 0 30px 20px; }
pre { font-family: 'SF Mono', Menlo, Monaco, monospace; font-size: 14.5px;
  line-height: 1.55; padding: 8px 30px 30px; white-space: pre-wrap; }
.info { color: #58a6ff; } .ok { color: #3fb950; } .warn { color: #d29922; font-weight:600; }
.err { color: #f85149; } .probe { color: #d29922; font-weight: 700; }
.muted { color: #8b949e; } .hl { color: #e6edf3; font-weight:600; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px;
  font-size:12px; font-weight:700; margin-left:8px; vertical-align:middle;}
.badge.up{background:#238636;color:#fff}.badge.down{background:#da3633;color:#fff}
/* bar chart */
.chart { padding: 10px 30px 34px; }
.bar-row { display:flex; align-items:center; margin:14px 0; }
.bar-label { width: 220px; font-family:'SF Mono',Menlo,monospace; font-size:14px; }
.bar-track { flex:1; background:#21262d; border-radius:6px; height:34px; position:relative;}
.bar-fill { height:100%; border-radius:6px; font-weight:700; font-size:14px; color:#fff;
  min-width:6px; }
.bar-val { position:absolute; top:0; height:34px; display:flex; align-items:center;
  padding-left:14px; font-weight:700; font-size:14px; color:#e6edf3; white-space:nowrap; }
.bar-fill.primary{background:linear-gradient(90deg,#1f6feb,#58a6ff);}
.bar-fill.secondary{background:linear-gradient(90deg,#238636,#3fb950);}
.bar-fill.reject{background:linear-gradient(90deg,#da3633,#f85149);}
.legend{color:#8b949e;font-size:13px;padding:0 30px 26px;}
"""


def colorize(line: str) -> str:
    """Wrap a captured text line in spans so it renders with terminal-like color."""
    e = html.escape(line)
    if "HEALTH TRANSITION" in e or "REJECTED" in e:
        return f'<span class="warn">{e}</span>'
    if "PROBE" in e:
        return f'<span class="probe">{e}</span>'
    if "FAILED" in e or "ERROR" in e:
        return f'<span class="err">{e}</span>'
    if " OK" in e or "-> anthropic" in e or "-> openai" in e:
        return f'<span class="ok">{e}</span>'
    if e.strip().startswith("INFO"):
        return f'<span class="info">{e}</span>'
    if e.strip().startswith(("=", "SCENARIO", "RESULT", "->")):
        return f'<span class="hl">{e}</span>'
    return f'<span class="muted">{e}</span>'


def page(title: str, subtitle: str, filename: str, body_html: str) -> None:
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div class="card">
  <div class="titlebar"><span class="dot r"></span><span class="dot y"></span>
    <span class="dot g"></span><span class="name">{html.escape(filename)}</span></div>
  <h1>{html.escape(title)}</h1>
  <div class="sub">{subtitle}</div>
  {body_html}
</div></body></html>"""
    path = os.path.join(OUT, filename)
    with open(path, "w") as f:
        f.write(doc)
    print("wrote", path)


def pre_from_lines(lines: list[str]) -> str:
    return "<pre>" + "\n".join(colorize(l) for l in lines) + "</pre>"


def read(path: str) -> list[str]:
    with open(path) as f:
        return [l.rstrip("\n") for l in f]


def slice_between(lines: list[str], start_re: str, end_re: str | None) -> list[str]:
    """Return the lines from the first match of start_re to just before end_re."""
    out, capturing = [], False
    for l in lines:
        if not capturing and re.search(start_re, l):
            capturing = True
        if capturing:
            if end_re and re.search(end_re, l) and out:
                break
            out.append(l)
    return out


def main() -> None:
    demo = read("/tmp/demo_output.txt")
    trace = read("/tmp/trace20_output.txt")

    # --- Image 1: the 20-request routing decision (answers the direct question)
    page(
        "Routing decision — 20 requests, primary UNHEALTHY",
        "Weights <span class='hl'>[primary = 5%, secondary = 95%]</span>. "
        "Smoothed Weighted Round-Robin picks each request deterministically: "
        "the single 5% probe lands on request #10, evenly spaced — not random, not front-loaded.",
        "01_routing_decision_20_requests.html",
        pre_from_lines(trace),
    )

    # --- Image 2: health transition (primary trips UNHEALTHY after x=3 failures)
    s2 = slice_between(demo, r"SCENARIO 2", r"SCENARIO 3")
    page(
        "Health detection — primary trips UNHEALTHY after x=3 failures",
        "Three consecutive failures flip the primary to "
        "<span class='badge down'>UNHEALTHY</span>. The router logs the transition "
        "and immediately recomputes weights to <span class='hl'>[5, 95]</span>. "
        "Failed calls fail fast to the caller (no cross-provider retry).",
        "02_health_transition_unhealthy.html",
        pre_from_lines([l for l in s2 if l.strip()]),
    )

    # --- Image 3: recovery — primary snaps back to HEALTHY / 100%
    s4 = slice_between(demo, r"SCENARIO 4", r"DEMONSTRATION COMPLETE")
    # Head up to (and including) the first UNHEALTHY transition, then the
    # recovery (HEALTHY) transition, then the tail summary — no duplicate lines.
    head = []
    for l in s4[:12]:
        head.append(l)
        if "HEALTH TRANSITION" in l and "UNHEALTHY" in l:
            break
    recovery = [l for l in s4 if "HEALTH TRANSITION" in l and "-> HEALTHY" in l]
    tail = s4[-6:]
    page(
        "Recovery — primary earns y=3 probes and snaps back to 100%",
        "While degraded, the 5% probe stream keeps sampling the primary. After "
        "<span class='hl'>3 consecutive successful probes</span> it flips back to "
        "<span class='badge up'>HEALTHY</span>, weights reset to "
        "<span class='hl'>[100, 0]</span>, and it reclaims all traffic.",
        "03_recovery_snap_back.html",
        pre_from_lines(head + ["    ... (5% probe stream continues sampling the primary) ..."]
                       + recovery + [""] + tail),
    )

    # --- Image 4: distribution bar chart (200-request degraded window)
    m = re.search(r"anthropic\(secondary\)': (\d+).*?ERROR': (\d+)",
                  next(l for l in demo if "RESULT over 200" in l and "ERROR" in l))
    sec, err = int(m.group(1)), int(m.group(2))
    total = sec + err
    sec_pct, err_pct = round(sec / total * 100), round(err / total * 100)
    chart = f"""
    <div class="chart">
      <div class="bar-row"><div class="bar-label">anthropic(secondary)<br><span class="muted">healthy · bulk</span></div>
        <div class="bar-track"><div class="bar-fill secondary" style="width:{sec_pct}%"></div>
          <div class="bar-val" style="left:{sec_pct}%; transform:translateX(-100%); color:#fff;">{sec} req · {sec_pct}%</div></div></div>
      <div class="bar-row"><div class="bar-label">openai(primary)<br><span class="muted">unhealthy · 5% probe</span></div>
        <div class="bar-track"><div class="bar-fill primary" style="width:{max(err_pct,3)}%"></div>
          <div class="bar-val" style="left:{max(err_pct,3)}%;">{err} req · {err_pct}%</div></div></div>
    </div>
    <div class="legend">Measured over 200 live requests in the degraded steady state.
      The split is <span class="hl">exact</span> (5% × 200 = 10), because SWRR is deterministic — not sampled.</div>
    """
    page(
        "Traffic distribution — degraded steady state (200 requests)",
        "Primary UNHEALTHY, secondary HEALTHY, p=5. Observed routing over 200 real requests:",
        "04_distribution_chart.html",
        chart,
    )


if __name__ == "__main__":
    main()
