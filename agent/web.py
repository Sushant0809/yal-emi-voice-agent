"""Borrower picker page, served by the Pipecat runner at http://localhost:7860/borrowers.

Pick who Asha calls, see their test verification details and past calls, clear
their history, then open the Pipecat Playground to start the call.
"""

import json
from datetime import date

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pipecat.runner.run import app
from pydantic import BaseModel

from agent.borrowers import load_borrowers
from agent.config import CONFIG
from agent.history import connect, recent_calls
from agent.phrases import lender
from agent.speech import date_to_words, rupees_to_words
from agent.use_borrower import ENV, set_borrower


class Selection(BaseModel):
    borrower_id: str


def _current() -> str:
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


@app.get("/api/borrowers")
async def list_borrowers():
    return {"current": _current(), "borrowers": _borrower_rows()}


@app.post("/api/borrower")
async def select_borrower(sel: Selection):
    if sel.borrower_id not in load_borrowers():
        raise HTTPException(404, "Unknown borrower")
    set_borrower(sel.borrower_id)
    return {"current": sel.borrower_id}


@app.post("/api/borrower/{borrower_id}/clear-history")
async def clear_history(borrower_id: str):
    with connect() as conn:
        n = conn.execute("DELETE FROM call_history WHERE borrower_id = ?", (borrower_id,)).rowcount
    return {"cleared": n}


@app.get("/borrowers", response_class=HTMLResponse)
async def borrowers_page():
    page = PAGE.replace("__DATA__", json.dumps(await list_borrowers())).replace("__LENDER__", lender().name)
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
const $ = id => document.getElementById(id);
function render(){
  const b = data.borrowers.find(x => x.id === data.current) || data.borrowers[0];
  $("sel").innerHTML = data.borrowers.map(x =>
    `<option value="${x.id}" ${x.id===b.id?"selected":""}>${x.name} · ${x.id} · ${x.language}</option>`).join("");
  $("facts").innerHTML = `<dt>Loan</dt><dd>${b.loan}</dd><dt>Amount</dt><dd>${b.amount}</dd>
    <dt>Status</dt><dd>${b.status}</dd><dt>Language</dt><dd>${b.language}</dd>`;
  $("verify").innerHTML = `To verify, say DOB <b>${b.dob}</b> or mobile ending <b>${b.last4}</b>`;
  $("hist").innerHTML = b.history.length
    ? b.history.map(h => `<li>${h.when} · <b>${h.outcome}</b>${h.ptp ? " · promised " + h.ptp : ""}</li>`).join("")
    : `<li class="muted">No previous calls. Asha starts fresh.</li>`;
}
async function refresh(){ data = await (await fetch("/api/borrowers")).json(); render(); }
$("sel").onchange = async e => {
  await fetch("/api/borrower", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({borrower_id: e.target.value})});
  await refresh(); $("msg").textContent = "Saved. The next call goes to this borrower.";
};
$("clear").onclick = async () => {
  const r = await (await fetch(`/api/borrower/${data.current}/clear-history`, {method:"POST"})).json();
  await refresh(); $("msg").textContent = `Cleared ${r.cleared} previous call(s).`;
};
render();
</script></body></html>"""
