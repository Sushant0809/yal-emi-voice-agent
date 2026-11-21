"""Data retention and erasure.

    uv run python -m agent.retention --dry-run      # what would be deleted now
    uv run python -m agent.retention                # delete it
    uv run python -m agent.retention erase LF-1001  # one borrower's data, everywhere (deletion request)

Recordings and transcripts hold the most sensitive data (voice, DOB, phone digits, loan
details), so they are kept for RETENTION_DAYS (default 30; 0 keeps them forever) and then
deleted. The bot runs the purge in the background at the start of every call. Also purged:
rotated bot logs, and identity-check rows (`verification_attempts`) older than the window.

Kept: the outcome record in call history (no transcript, no audio), because cross-call memory
and reporting need it. `erase` removes that too, for one borrower. Every deletion is written to
the audit log.
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

from agent import audit
from agent.config import CONFIG
from agent.history import connect


def _call_time(path: Path) -> datetime:
    """When the call happened: from the call id in the file name (UTC), else the file's mtime."""
    try:
        return datetime.strptime(path.stem[:15], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def _call_files(outcomes: Path) -> list[Path]:
    return [*sorted((outcomes / "recordings").glob("*.wav")),
            *sorted((outcomes / "transcripts").glob("*.txt")),
            *sorted((outcomes / "transcripts").glob("*.json"))]


def purge(days: int | None = None, now: datetime | None = None, outcomes: Path | None = None,
          db_path: Path | None = None, dry_run: bool = False, audit_path: Path | None = None) -> dict:
    """Delete call data older than `days`. Returns what was (or would be) deleted."""
    days = CONFIG.retention_days if days is None else days
    outcomes = Path(outcomes or CONFIG.outcomes_dir)
    if days <= 0:
        return {"files": [], "verification_rows": 0, "disabled": True}
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    old = [p for p in _call_files(outcomes) if _call_time(p) < cutoff]
    # Rotated logs only (loguru names them bot.<timestamp>.log); never the live bot.log.
    old += [p for p in sorted(outcomes.glob("bot.*.log"))
            if datetime.fromtimestamp(p.stat().st_mtime, UTC) < cutoff]
    with connect(db_path) as conn:
        where = "WHERE created_at < ?", (cutoff.isoformat(timespec="seconds"),)
        rows = conn.execute(f"SELECT COUNT(*) FROM verification_attempts {where[0]}", where[1]).fetchone()[0]
        if not dry_run and rows:
            conn.execute(f"DELETE FROM verification_attempts {where[0]}", where[1])
    if not dry_run:
        for p in old:
            p.unlink(missing_ok=True)
        if old or rows:
            audit.record("retention", "retention_delete", path=audit_path, days=days,
                         recordings=sum(p.suffix == ".wav" for p in old),
                         transcripts=sum(p.suffix in (".txt", ".json") for p in old),
                         logs=sum(p.suffix == ".log" for p in old), verification_rows=rows)
    return {"files": [str(p) for p in old], "verification_rows": rows, "cutoff": cutoff.isoformat()}


def purge_in_background() -> None:
    """For the bot: never let retention errors touch a call."""
    try:
        result = purge()
        if result["files"] or result["verification_rows"]:
            logger.info(f"Retention: deleted {len(result['files'])} files and "
                        f"{result['verification_rows']} identity-check rows older than {CONFIG.retention_days} days")
    except Exception as e:
        logger.warning(f"Retention purge failed (call unaffected): {e}")


def erase_borrower(borrower_id: str, actor: str, outcomes: Path | None = None, db_path: Path | None = None,
                   audit_path: Path | None = None) -> dict:
    """Everything held about one borrower: call history, identity checks, transcripts, recordings.
    The audit entry is written first: if it can't be, nothing is deleted."""
    outcomes = Path(outcomes or CONFIG.outcomes_dir)
    with connect(db_path) as conn:
        call_ids = {r[0] for r in conn.execute("SELECT call_id FROM call_history WHERE borrower_id = ?", (borrower_id,))}
    # Calls whose outcome was never saved to history are still findable from their transcript.
    for t in (outcomes / "transcripts").glob("*.json"):
        try:
            if (json.loads(t.read_text()).get("outcome") or {}).get("borrower_id") == borrower_id:
                call_ids.add(t.stem)
        except (OSError, ValueError):
            continue
    files = [p for p in _call_files(outcomes) if p.stem in call_ids]
    audit.record(actor, "erase_borrower", borrower_id, path=audit_path, calls=len(call_ids), files=len(files))
    for p in files:
        p.unlink(missing_ok=True)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM verification_attempts WHERE borrower_id = ?", (borrower_id,))
        rows = conn.execute("DELETE FROM call_history WHERE borrower_id = ?", (borrower_id,)).rowcount
    return {"calls": len(call_ids), "files": len(files), "history_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["purge", "erase"], default="purge")
    parser.add_argument("borrower_id", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--days", type=int, help=f"Override RETENTION_DAYS ({CONFIG.retention_days})")
    args = parser.parse_args()
    if args.command == "erase":
        if not args.borrower_id:
            parser.error("erase needs a borrower id")
        print(erase_borrower(args.borrower_id.upper(), audit.cli_actor()))
        return
    result = purge(args.days, dry_run=args.dry_run)
    if result.get("disabled"):
        print("Retention is off (RETENTION_DAYS=0).")
        return
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} {len(result['files'])} files and {result['verification_rows']} identity-check rows "
          f"from before {result['cutoff'][:10]}.")
    for f in result["files"][:20]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
