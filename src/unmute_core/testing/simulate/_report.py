"""The summary of a run directory, and two runs side by side.

A run directory holds ``calls/<scenario>/<nn>.jsonl``, one file per call, and
``run.json``. Only a call with its checks written counts. A call that is not scored (see
``scored``) is counted, never in a pass rate.
"""

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

LEFT_OUT = {
    "environment": "the endpoint failed (rate limit, 5xx, timeout)",
    "harness": "the simulator broke",
    "judge": "the judge did not answer",
    "caller": "the caller left its brief",
}
"""Why a call is not scored, in words."""


def _pct(values: Sequence[float], q: float) -> float | None:
    """Return the ``q`` quantile of the values, or None for none."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _median(values: Sequence[float]) -> float | None:
    """Return the median of the values, or None for none."""
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 0) -> str:
    """Return a number for a table cell."""
    return "-" if value is None else f"{value:.{digits}f}"


def scored(call: dict[str, Any]) -> bool:
    """Return whether a loaded call counts towards the pass rates.

    A call the endpoint or the simulator broke, that the judge never answered, or whose
    simulated caller left its brief is not evidence about the agent: it is counted, not
    scored.
    """
    return bool(call["checks"].get("valid", True))


def wrong_saves(calls: Iterable[dict[str, Any]]) -> int:
    """Return how many saves made as calls ended should not have been.

    Counted apart from pass and fail: one wrong save is a caller's answer kept or
    confirmed against what they said, whatever else the call got right.
    """
    return sum(
        len(result.get("evidence") or [])
        for call in calls
        for result in call["checks"]["results"]
        if result["id"] == "no_wrong_call_end_save"
    )


def run_meta(out: Path) -> dict[str, Any]:
    """Return what a run directory's ``run.json`` says, or nothing."""
    path = out / "run.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_records(path: Path) -> list[dict[str, Any]]:
    """Return the records of one call file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_calls(out: Path) -> list[dict[str, Any]]:
    """Return every call of a run directory that has its checks."""
    calls = []
    for path in sorted((out / "calls").glob("*/*.jsonl")):
        records = read_records(path)
        checks = next((r for r in records if r.get("type") == "checks"), None)
        if checks is None:
            continue
        events = [r for r in records if r.get("type") == "event"]
        calls.append(
            {
                "path": str(path),
                "scenario": checks["scenario"],
                "checks": checks,
                "call": next(r for r in records if r.get("type") == "call"),
                "events": events,
                "llm": [e for e in events if e["kind"] == "llm"],
            }
        )
    return calls


def _turn_latencies(events: list[dict[str, Any]]) -> list[float]:
    """Return, per caller line, the seconds until the agent's last line before the next."""
    latencies: list[float] = []
    caller_t: float | None = None
    last_agent: float | None = None
    for e in events:
        if e["kind"] == "caller_said":
            if caller_t is not None and last_agent is not None and last_agent > caller_t:
                latencies.append(last_agent - caller_t)
            caller_t, last_agent = float(e["t"]), None
        elif e["kind"] == "agent_said":
            last_agent = float(e["t"])
    if caller_t is not None and last_agent is not None and last_agent > caller_t:
        latencies.append(last_agent - caller_t)
    return latencies


def _scenario_row(scenario: str, rows: list[dict[str, Any]], left_out: int) -> dict[str, Any]:
    """Return one scenario's line of the summary."""
    passed = sum(1 for r in rows if r["checks"]["passed"])
    failing = Counter(
        res["id"]
        for r in rows
        for res in r["checks"]["results"]
        if not res["passed"] and res["core"]
    )
    return {
        "scenario": scenario,
        "calls": len(rows),
        "left_out": left_out,
        "passed": passed,
        "pass_rate": passed / len(rows) if rows else 0.0,
        "top_failures": failing.most_common(4),
        "requests_median": _median([len(r["llm"]) for r in rows]) or 0,
        "turns_median": _median([r["call"].get("turns", 0) for r in rows]) or 0,
    }


def summarize(out: Path) -> dict[str, Any]:
    """Return the summary of a run directory: pass rates, failures by check, latencies."""
    all_calls = load_calls(out)
    calls = [c for c in all_calls if scored(c)]
    unscored = [c for c in all_calls if not scored(c)]
    per_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        per_scenario[call["scenario"]].append(call)
    left_out = Counter(c["scenario"] for c in unscored)
    for scenario in left_out:
        per_scenario.setdefault(scenario, [])
    check_fail: Counter[str] = Counter()
    category_fail: Counter[tuple[str, str]] = Counter()
    check_core: dict[str, bool] = {}
    for call in calls:
        for result in call["checks"]["results"]:
            check_core[result["id"]] = result["core"]
            if not result["passed"]:
                check_fail[result["id"]] += 1
                for category in (result.get("category") or "unlabelled").split(","):
                    category_fail[(result["id"], category)] += 1
    requests = [len(c["llm"]) for c in all_calls]
    latencies = [r["latency_ms"] for c in all_calls for r in c["llm"]]
    call_llm_ms = [sum(r["latency_ms"] for r in c["llm"]) for c in all_calls]
    turn_lat = [x for c in all_calls for x in _turn_latencies(c["events"])]
    retried = sum(1 for c in all_calls if any(e["kind"] == "llm_retry" for e in c["events"]))
    meta = run_meta(out)
    return {
        "framework": meta.get("framework"),
        "models": meta.get("models") or {},
        "calls": len(calls),
        "calls_run": len(all_calls),
        "left_out": len(unscored),
        "left_out_reasons": dict(Counter(c["checks"].get("left_out") or "?" for c in unscored)),
        "calls_with_rate_limit_retry": retried,
        "passed": sum(1 for c in calls if c["checks"]["passed"]),
        "wrong_saves": wrong_saves(calls),
        "scenarios": [
            _scenario_row(scenario, rows, left_out[scenario])
            for scenario, rows in sorted(per_scenario.items())
        ],
        "check_failures": [
            {
                "check": k,
                "core": check_core.get(k, False),
                "calls": v,
                "share": v / len(calls) if calls else 0.0,
            }
            for k, v in check_fail.most_common()
        ],
        "category_failures": [
            {"check": k[0], "category": k[1], "calls": v} for k, v in category_fail.most_common()
        ],
        "metrics": {
            "requests_per_call_median": _median(requests),
            "requests_per_call_p90": _pct(requests, 0.9),
            "request_latency_ms_median": _median(latencies),
            "request_latency_ms_p90": _pct(latencies, 0.9),
            "llm_ms_per_call_median": _median(call_llm_ms),
            "llm_ms_per_call_p90": _pct(call_llm_ms, 0.9),
            "turn_latency_s_median": _median(turn_lat),
            "turn_latency_s_p90": _pct(turn_lat, 0.9),
        },
    }


def render(summary: dict[str, Any], title: str) -> str:
    """Return the summary as Markdown."""
    lines = [f"# {title}", ""]
    if summary.get("framework"):
        lines += [f"Framework: **{summary['framework']}**."]
    lines += [f"- {role}: `{label}`" for role, label in summary.get("models", {}).items()]
    total, passed = summary["calls"], summary["passed"]
    reasons = ", ".join(
        f"{n} {LEFT_OUT.get(reason, reason)}" for reason, n in summary["left_out_reasons"].items()
    )
    lines += [
        "",
        f"**{passed}/{total} scored calls passed every core check "
        f"({passed / total * 100 if total else 0:.0f}%).** {summary['calls_run']} calls ran; "
        f"{summary['left_out']} were left out ({reasons or 'none'}). "
        f"{summary['calls_with_rate_limit_retry']} calls had an agent request retried after "
        "a rate limit.",
        "",
        f"Wrong saves at call end: **{summary['wrong_saves']}**.",
        "",
        "## Pass rate per scenario",
        "",
        "| scenario | scored calls | passed | rate | left out | model requests (median) "
        "| caller turns (median) | top failing core checks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["scenarios"]:
        tops = ", ".join(f"{k} x{v}" for k, v in row["top_failures"]) or "-"
        lines.append(
            f"| {row['scenario']} | {row['calls']} | {row['passed']} "
            f"| {row['pass_rate'] * 100:.0f}% | {row['left_out']} | {row['requests_median']} "
            f"| {row['turns_median']} | {tops} |"
        )
    lines += ["", "## Failures by check", "", "| check | core | calls failing | share |"]
    lines.append("|---|---|---|---|")
    lines += [
        f"| {row['check']} | {'yes' if row['core'] else 'extra'} | {row['calls']} "
        f"| {row['share'] * 100:.0f}% |"
        for row in summary["check_failures"]
    ]
    lines += ["", "## Failures by category", "", "| check | category | calls |", "|---|---|---|"]
    lines += [
        f"| {row['check']} | {row['category']} | {row['calls']} |"
        for row in summary["category_failures"]
    ]
    m = summary["metrics"]
    lines += [
        "",
        "## Agent model requests and latency",
        "",
        "| metric | median | p90 |",
        "|---|---|---|",
        f"| model requests per call | {_fmt(m['requests_per_call_median'])} "
        f"| {_fmt(m['requests_per_call_p90'])} |",
        f"| latency per request (ms) | {_fmt(m['request_latency_ms_median'])} "
        f"| {_fmt(m['request_latency_ms_p90'])} |",
        f"| model time per call (ms) | {_fmt(m['llm_ms_per_call_median'])} "
        f"| {_fmt(m['llm_ms_per_call_p90'])} |",
        f"| caller line to agent's last line (s) | {_fmt(m['turn_latency_s_median'], 1)} "
        f"| {_fmt(m['turn_latency_s_p90'], 1)} |",
    ]
    return "\n".join(lines) + "\n"


def write_summary(out: Path) -> dict[str, Any]:
    """Write ``summary.json`` and ``summary.md`` into a run directory, and return the summary."""
    summary = summarize(out)
    label = run_meta(out).get("label")
    title = f"Call simulator: {out.name}" + (f" ({label})" if label else "")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text(render(summary, title))
    return summary


def fisher_p(a_fail: int, a_n: int, b_fail: int, b_n: int) -> float:
    """Return the two-sided Fisher exact p of failure counts ``a_fail/a_n`` vs ``b_fail/b_n``."""
    fails, total = a_fail + b_fail, a_n + b_n
    if total == 0:
        return 1.0

    def prob(x: int) -> float:
        """Return the chance of ``x`` failures on side a, the totals fixed."""
        return math.comb(a_n, x) * math.comb(b_n, fails - x) / math.comb(total, fails)

    observed = prob(a_fail)
    low, high = max(0, fails - b_n), min(a_n, fails)
    return min(1.0, sum(prob(x) for x in range(low, high + 1) if prob(x) <= observed * (1 + 1e-9)))


def _side(out: Path, scenarios: set[str] | None) -> dict[str, Any]:
    """Return one run's numbers for ``compare``."""
    summary = summarize(out)
    calls = [
        c
        for c in load_calls(out)
        if scored(c) and (scenarios is None or c["scenario"] in scenarios)
    ]
    fails: Counter[str] = Counter(
        result["id"]
        for call in calls
        for result in call["checks"]["results"]
        if not result["passed"]
    )
    return {
        "per": {
            s["scenario"]: s
            for s in summary["scenarios"]
            if scenarios is None or s["scenario"] in scenarios
        },
        "fails": fails,
        "n": len(calls),
        "passed": sum(1 for c in calls if c["checks"]["passed"]),
        "requests": _median([len(c["llm"]) for c in calls]),
    }


def compare(base: Path, other: Path, *, scenarios: set[str] | None = None) -> str:
    """Return Markdown: two runs' pass rates and check failures side by side.

    Each check's failure counts get a two-sided Fisher exact p, so a change from a handful
    of calls is not read as a trend.
    """
    a, b = _side(base, scenarios), _side(other, scenarios)

    def share(side: dict[str, Any], count: int) -> str:
        """Return a count with its share of the side's scored calls."""
        return f"{count} ({count / max(side['n'], 1) * 100:.0f}%)"

    lines = [
        f"| | {base.name} | {other.name} |",
        "|---|---|---|",
        f"| scored calls | {a['n']} | {b['n']} |",
        f"| passed every core check | {share(a, a['passed'])} | {share(b, b['passed'])} |",
        f"| model requests per call (median) | {_fmt(a['requests'])} | {_fmt(b['requests'])} |",
        "",
        f"| scenario | {base.name} | {other.name} |",
        "|---|---|---|",
    ]
    for name in sorted(set(a["per"]) | set(b["per"])):
        cells = [
            f"{row['passed']}/{row['calls']}" if (row := side["per"].get(name)) else "-"
            for side in (a, b)
        ]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        f"| check (calls failing) | {base.name} | {other.name} | Fisher p |",
        "|---|---|---|---|",
    ]
    for check in sorted(
        set(a["fails"]) | set(b["fails"]), key=lambda k: -(a["fails"][k] + b["fails"][k])
    ):
        p = fisher_p(a["fails"][check], a["n"], b["fails"][check], b["n"])
        lines.append(
            f"| {check} | {share(a, a['fails'][check])} | {share(b, b['fails'][check])} | {p:.3f} |"
        )
    return "\n".join(lines) + "\n"
