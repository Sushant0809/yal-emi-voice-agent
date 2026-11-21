"""Borrower picker page, served by the Pipecat runner at http://localhost:7860/borrowers.

Pick who Asha calls, see their test verification details and past calls, clear
their history, then open the Pipecat Playground to start the call.

The page shows verification secrets (DOB, mobile digits), so it and its API need a token:
ADMIN_TOKEN from .env, or a random one printed at start-up. Open /borrowers?token=<token>.
"""

import html
import json
import secrets
from datetime import date

from fastapi import Depends, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from loguru import logger
from pipecat.runner.run import app
from pydantic import BaseModel

from agent import audit
from agent.borrowers import load_borrowers
from agent.config import CONFIG
from agent.history import clear_borrower, connect, recent_calls
from agent.phrases import lender
from agent.speech import date_to_words, rupees_to_words
from agent.use_borrower import ENV, set_borrower

ADMIN_TOKEN = CONFIG.admin_token or secrets.token_urlsafe(16)
if not CONFIG.admin_token and not CONFIG.admin_tokens:
    logger.info(f"Borrower picker: http://localhost:7860/borrowers?token={ADMIN_TOKEN}")


def _tokens() -> dict[str, str]:
    """token -> who holds it. ADMIN_TOKENS names each person; the single ADMIN_TOKEN is "admin"."""
    named = dict(reversed(pair.split(":", 1)) for pair in CONFIG.admin_tokens.split(",") if ":" in pair)
    return {**({ADMIN_TOKEN: "admin"} if CONFIG.admin_token or not named else {}), **named}


def require_token(x_admin_token: str | None = Header(None), token: str | None = Query(None)) -> str:
    """The caller's name, for the audit log. 401 without a valid token."""
    given = (x_admin_token or token or "").encode()
    for known, who in _tokens().items():
        if secrets.compare_digest(given, known.encode()):
            return f"web:{who}"
    raise HTTPException(401, "Missing or wrong admin token. Open /borrowers?token=<ADMIN_TOKEN>.")


class Selection(BaseModel):
    borrower_id: str


def _current() -> str:
    if not ENV.exists():  # e.g. in CI, or before .env is created
        return CONFIG.default_borrower_id
    for line in ENV.read_text().splitlines():
        if line.startswith("DEFAULT_BORROWER_ID="):
            return line.split("=", 1)[1].strip()
    return CONFIG.default_borrower_id


def _borrower_rows() -> list[dict]:
    rows = []
    for b in load_borrowers().values():
        rows.append({
            "id": b.borrower_id,
            "name": b.name,
            "loan": b.loan_type,
            "language": {"en": "English", "hi": "Hindi"}.get(b.preferred_language, b.preferred_language),
            "dob": date.fromisoformat(b.dob).strftime("%d %b %Y"),
            "last4": b.phone_last4,
            "amount": rupees_to_words(b.overdue_amount if b.is_overdue else b.emi_amount),
            "status": (f"{b.days_overdue} days overdue" if b.is_overdue
                       else f"due {date_to_words(date.fromisoformat(b.due_date))}"),
            "history": [
                {"outcome": c["outcome"], "when": c["created_at"][:16].replace("T", " "),
                 "ptp": c["promise_to_pay_date"]}
                for c in recent_calls(b.borrower_id, limit=5)
            ],
        })
    return rows


# Every endpoint is audited: the page shows every borrower's DOB and mobile digits. The audit
# entry is written before the data is returned or changed, so a failed write blocks the request.


@app.get("/api/borrowers")
async def list_borrowers(actor: str = Depends(require_token)):
    audit.record(actor, "view_borrowers", count=len(load_borrowers()))
    return {"current": _current(), "borrowers": _borrower_rows()}


@app.post("/api/borrower")
async def select_borrower(sel: Selection, actor: str = Depends(require_token)):
    if sel.borrower_id not in load_borrowers():
        raise HTTPException(404, "Unknown borrower")
    audit.record(actor, "select_borrower", sel.borrower_id)
    set_borrower(sel.borrower_id)
    return {"current": sel.borrower_id}


@app.post("/api/borrower/{borrower_id}/clear-history")
async def clear_history(borrower_id: str, actor: str = Depends(require_token)):
    if borrower_id not in load_borrowers():
        raise HTTPException(404, "Unknown borrower")
    audit.record(actor, "clear_history", borrower_id)
    with connect() as conn:
        n = clear_borrower(conn, borrower_id)  # its failed identity checks too, which unlocks it
    return {"cleared": n}


def _script_json(value) -> str:
    """JSON safe to embed in a <script>: "</script>" in a name or note can't end the block."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


@app.get("/borrowers", response_class=HTMLResponse)
async def borrowers_page(actor: str = Depends(require_token), x_admin_token: str | None = Header(None),
                         token: str | None = Query(None)):
    audit.record(actor, "view_borrowers", count=len(load_borrowers()))
    data = {"current": _current(), "borrowers": _borrower_rows()}
    # The page's API calls reuse the token it was opened with, so they're audited as the same person.
    page = (PAGE.replace("__DATA__", _script_json(data)).replace("__TOKEN__", _script_json(x_admin_token or token))
            .replace("__LENDER__", html.escape(lender().name)))
    return HTMLResponse(page)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__LENDER__ · Test calls</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e4df;--accent:#1f6f5c;--warn:#a4441c}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecece8;--muted:#9a9a94;--line:#33332f;--accent:#5fbf9f;--warn:#e38b62}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:560px;margin:40px auto;padding:0 16px}
h1{font-size:20px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:14px}
label{font-size:13px;color:var(--muted);display:block;margin-bottom:6px}
select{width:100%;font:inherit;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin:14px 0 0}dt{color:var(--muted)}dd{margin:0;font-weight:500}
.verify{background:var(--bg);border-radius:8px;padding:10px 12px;margin-top:14px;font-size:14px}
.verify b{font-variant-numeric:tabular-nums}
.hist{font-size:14px;margin:8px 0 0;padding:0;list-style:none}.hist li{padding:4px 0;border-top:1px solid var(--line)}
.muted{color:var(--muted)}.row{display:flex;gap:10px;flex-wrap:wrap}
button,a.btn{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;text-decoration:none}
a.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.danger{color:var(--warn)}#msg{font-size:13px;color:var(--accent);min-height:18px;margin-top:8px}
</style></head><body><main>
<h1>Who should Asha call?</h1>
<p class="sub">Pick a borrower, then start the call. The bot reads this at the start of every call; no restart needed.</p>
<div class="card">
  <label for="sel">Borrower</label>
  <select id="sel"></select>
  <dl id="facts"></dl>
  <div class="verify" id="verify"></div>
</div>
<div class="card">
  <div class="row" style="justify-content:space-between;align-items:center">
    <strong>Previous calls</strong><button class="danger" id="clear">Clear history</button>
  </div>
  <ul class="hist" id="hist"></ul>
</div>
<div class="row"><a class="btn primary" href="/client/">Start call in Playground →</a></div>
<div id="msg"></div>
</main>
<script>
let data = __DATA__;
const TOKEN = __TOKEN__;
const $ = id => document.getElementById(id);
// Every value from the server is escaped before it goes into innerHTML.
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const api = (path, opts = {}) => fetch(path, {...opts, headers: {...(opts.headers || {}), "X-Admin-Token": TOKEN}});
function render(){
  const b = data.borrowers.find(x => x.id === data.current) || data.borrowers[0];
  $("sel").innerHTML = data.borrowers.map(x =>
    `<option value="${esc(x.id)}" ${x.id===b.id?"selected":""}>${esc(x.name)} · ${esc(x.id)} · ${esc(x.language)}</option>`).join("");
  $("facts").innerHTML = `<dt>Loan</dt><dd>${esc(b.loan)}</dd><dt>Amount</dt><dd>${esc(b.amount)}</dd>
    <dt>Status</dt><dd>${esc(b.status)}</dd><dt>Language</dt><dd>${esc(b.language)}</dd>`;
  $("verify").innerHTML = `To verify, say DOB <b>${esc(b.dob)}</b> or mobile ending <b>${esc(b.last4)}</b>`;
  $("hist").innerHTML = b.history.length
    ? b.history.map(h => `<li>${esc(h.when)} · <b>${esc(h.outcome)}</b>${h.ptp ? " · promised " + esc(h.ptp) : ""}</li>`).join("")
    : `<li class="muted">No previous calls. Asha starts fresh.</li>`;
}
async function refresh(){ data = await (await api("/api/borrowers")).json(); render(); }
$("sel").onchange = async e => {
  await api("/api/borrower", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({borrower_id: e.target.value})});
  await refresh(); $("msg").textContent = "Saved. The next call goes to this borrower.";
};
$("clear").onclick = async () => {
  const r = await (await api(`/api/borrower/${encodeURIComponent(data.current)}/clear-history`, {method:"POST"})).json();
  await refresh(); $("msg").textContent = `Cleared ${r.cleared} previous call(s).`;
};
render();
</script></body></html>"""
