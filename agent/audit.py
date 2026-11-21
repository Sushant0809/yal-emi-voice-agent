"""Audit log: who accessed, changed or deleted borrower data, and when.

    uv run python -m agent.audit                    # last 20 entries
    uv run python -m agent.audit --borrower LF-1001 # everything touching one borrower
    uv run python -m agent.audit --verify           # check nothing was edited or removed

Append-only JSON lines in outcomes/audit.jsonl. Each entry carries the hash of the entry
before it, so editing, deleting or reordering a line breaks the chain and ``--verify``
says where. (Tamper-evident, not tamper-proof: in production the log would also be shipped
to write-once storage.) Entries never contain PII themselves: details are redacted, and
borrowers are identified by their internal id.

Who: the admin page's named token (ADMIN_TOKENS), ``cli:<os user>`` for the command-line
tools, ``agent`` for the voice agent releasing loan details to a verified caller, and
``retention`` for automatic deletion.
"""

import argparse
import fcntl
import getpass
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from agent.config import CONFIG
from agent.redact import redact

AUDIT_PATH = CONFIG.outcomes_dir / "audit.jsonl"
GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cli_actor() -> str:
    try:
        return f"cli:{getpass.getuser()}"
    except Exception:
        return "cli:unknown"


def record(actor: str, action: str, borrower_id: str | None = None, path: Path | None = None, **details) -> dict:
    """Append one entry. Raises if it can't be written: callers that change data should fail
    closed (no audit, no action); the live call path catches it instead."""
    path = Path(path or AUDIT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # one writer at a time, so the chain stays linear
        try:
            f.seek(0)
            lines = [line for line in f.read().splitlines() if line.strip()]
            prev = json.loads(lines[-1])["hash"] if lines else GENESIS
            entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "actor": actor, "action": action,
                     "borrower_id": borrower_id, "details": redact(details), "prev": prev}
            entry["hash"] = _digest(entry)
            f.seek(0, 2)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return entry


def verify(path: Path | None = None) -> tuple[bool, int, str]:
    """(ok, entries checked, message). Fails at the first entry whose hash or link is wrong."""
    path = Path(path or AUDIT_PATH)
    if not path.exists():
        return True, 0, "No audit log yet."
    prev = GENESIS
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except ValueError:
            return False, i - 1, f"Line {i} is not valid JSON."
        if entry.get("prev") != prev:
            return False, i - 1, f"Line {i} doesn't follow line {i - 1} (an entry was removed or reordered)."
        if entry.get("hash") != _digest(entry):
            return False, i - 1, f"Line {i} was modified after it was written."
        prev = entry["hash"]
    return True, len(lines), f"OK: {len(lines)} entries, chain intact."


def entries(path: Path | None = None) -> list[dict]:
    path = Path(path or AUDIT_PATH)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--borrower")
    parser.add_argument("--last", type=int, default=20)
    args = parser.parse_args()
    if args.verify:
        ok, _, message = verify()
        print(message)
        raise SystemExit(0 if ok else 1)
    rows = [e for e in entries() if not args.borrower or e["borrower_id"] == args.borrower]
    for e in rows[-args.last:]:
        print(f"{e['ts']}  {e['actor']:<18} {e['action']:<24} {e['borrower_id'] or '-':<8} "
              f"{json.dumps(e['details'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
