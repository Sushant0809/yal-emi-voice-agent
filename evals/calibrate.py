"""Judge calibration: how often does the LLM judge agree with a human reviewer?

    # 1. Make a labelling sheet from judged calls (online report, or an offline eval run)
    uv run python -m evals.calibrate template results/online/all.jsonl --sample 30
    # 2. A human fills in the blank columns in results/calibration/labels.csv, reading each transcript
    # 3. Compare
    uv run python -m evals.calibrate compare results/calibration/labels.csv results/online/all.jsonl

Per criterion it reports the number of calls both labelled, raw agreement and Cohen's kappa
(agreement corrected for chance), and for the 1-5 scores also agreement within one point and
quadratic-weighted kappa (a 4 vs 5 disagreement costs less than 1 vs 5). For yes/no criteria it
counts each direction of disagreement, because "the judge missed a violation" and "the judge
flagged a clean call" have very different costs.

Rule of thumb (Landis and Koch): kappa below 0.4 means don't trust the judge on that criterion
yet (fix the rubric, then re-label); 0.6 or more is substantial agreement.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from agent.config import ROOT

OUT_DIR = ROOT / "results" / "calibration"
SCALE = ["tone", "empathy", "naturalness"]      # 1-5
BINARY = ["guardrail_violation", "outcome_correct"]  # yes / no
CRITERIA = SCALE + BINARY


# --- Agreement statistics ---------------------------------------------------------------------


def cohens_kappa(a: list, b: list) -> float | None:
    """Unweighted Cohen's kappa. None when undefined (both raters used a single, identical class)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def weighted_kappa(a: list[int], b: list[int], categories: range = range(1, 6)) -> float | None:
    """Quadratic-weighted kappa for ordinal scores."""
    n, cats = len(a), list(categories)
    if n == 0:
        return None
    span = (len(cats) - 1) ** 2
    w = {(i, j): (i - j) ** 2 / span for i in cats for j in cats}
    observed = Counter(zip(a, b, strict=True))
    ca, cb = Counter(a), Counter(b)
    num = sum(w[i, j] * observed[i, j] for i in cats for j in cats)
    den = sum(w[i, j] * ca[i] * cb[j] / n for i in cats for j in cats)
    return None if den == 0 else 1 - num / den


def band(kappa: float | None) -> str:
    if kappa is None:
        return "n/a (only one class seen)"
    for limit, label in [(0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if kappa <= limit:
            return label
    return "almost perfect"


# --- Loading ----------------------------------------------------------------------------------


def _bool(value) -> bool | None:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n"):
        return False
    return None


def _score(value) -> int | None:
    try:
        v = int(str(value).strip())
    except ValueError:
        return None
    return v if 1 <= v <= 5 else None


def parse(criterion: str, value) -> int | bool | None:
    return _score(value) if criterion in SCALE else _bool(value)


def load_judge(path: Path) -> dict[str, dict]:
    """Judge verdicts keyed by id, from an online report (call_id) or an eval run (scenario#run)."""
    verdicts = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("judge"):
            continue
        key = r.get("call_id") or f"{r['scenario']}#{r['run']}"
        verdicts[key] = {c: parse(c, r["judge"].get(c)) for c in CRITERIA if c in r["judge"]}
        verdicts[key]["_transcript"] = r.get("transcript", "")
    return verdicts


def load_labels(path: Path) -> dict[str, dict]:
    with path.open(newline="") as f:
        return {row["id"]: {c: parse(c, row.get(c, "")) for c in CRITERIA} for row in csv.DictReader(f)}


# --- Commands ---------------------------------------------------------------------------------


def write_template(judged: Path, sample: int | None, seed: int = 0) -> Path:
    """A sheet for a human: one row per judged call, criterion columns left blank. The judge's
    own verdicts are deliberately not in it, so the reviewer isn't anchored by them."""
    verdicts = load_judge(judged)
    ids = sorted(verdicts)
    if sample and sample < len(ids):
        ids = sorted(random.Random(seed).sample(ids, sample))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "labels.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "transcript", *CRITERIA, "notes"])
        for i in ids:
            t = verdicts[i]["_transcript"]
            w.writerow([i, t if len(t) < 120 else "(inline transcript in the judged file)", *[""] * len(CRITERIA), ""])
    return path


def compare(labels: dict[str, dict], verdicts: dict[str, dict]) -> str:
    lines = ["| Criterion | Calls | Agreement | Cohen's kappa | | Within 1 pt | Weighted kappa |",
             "|---|---|---|---|---|---|---|"]
    disagreements = []
    for c in CRITERIA:
        pairs = [(h[c], verdicts[i][c]) for i, h in labels.items()
                 if i in verdicts and h.get(c) is not None and verdicts[i].get(c) is not None]
        if not pairs:
            lines.append(f"| {c} | 0 | - | - | | | |")
            continue
        human, judge = zip(*pairs, strict=True)
        k = cohens_kappa(list(human), list(judge))
        agree = sum(x == y for x, y in pairs) / len(pairs)
        row = f"| {c} | {len(pairs)} | {agree:.0%} | {'-' if k is None else f'{k:.2f}'} | {band(k)} |"
        if c in SCALE:
            within = sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs)
            wk = weighted_kappa(list(human), list(judge))
            row += f" {within:.0%} | {'-' if wk is None else f'{wk:.2f}'} |"
        else:
            missed = sum(h and not j for h, j in pairs)
            extra = sum(j and not h for h, j in pairs)
            row += " | |"
            disagreements.append(f"- {c}: human yes / judge no **{missed}**, judge yes / human no **{extra}**")
        lines.append(row)
    unmatched = sorted(set(labels) - set(verdicts))
    out = "\n".join(lines)
    if disagreements:
        out += "\n\nDisagreement direction (yes/no criteria):\n" + "\n".join(disagreements)
    if unmatched:
        out += f"\n\n{len(unmatched)} labelled ids had no judge verdict: {', '.join(unmatched[:5])}"
    return out + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="Write a blank labelling sheet for judged calls")
    t.add_argument("judged", type=Path, help="results/online/<date>.jsonl or results/<run>/runs.jsonl")
    t.add_argument("--sample", type=int, help="Label a random sample of this many calls")
    c = sub.add_parser("compare", help="Agreement and Cohen's kappa, human vs judge")
    c.add_argument("labels", type=Path)
    c.add_argument("judged", type=Path)
    args = parser.parse_args()

    if args.cmd == "template":
        print(f"Wrote {write_template(args.judged, args.sample)}. Fill in {', '.join(CRITERIA)} "
              "(1-5 for scores, yes/no for the rest); leave a cell blank to skip it.")
        return
    text = compare(load_labels(args.labels), load_judge(args.judged))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.labels.stem}-vs-{args.judged.parent.name}-{args.judged.stem}.md"
    out.write_text(f"# Judge calibration: {args.labels.name} vs {args.judged}\n\n{text}")
    print(text + f"\nSaved to {out}")


if __name__ == "__main__":
    main()
