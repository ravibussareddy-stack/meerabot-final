import json
import asyncio
import sys
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler
from pipeline import run_pipeline, PipelineResult
from config import TELEGRAM_BOT_TOKEN, MEERA_CHAT_ID, WEBHOOK_SECRET


def _score_bar(value: int, width: int = 10) -> str:
    filled = round(value * width / 10)
    return "█" * filled + "░" * (width - filled)


def _format_scores(r: PipelineResult) -> str:
    s = r.scores
    lines = [
        f"📊 *Note scored {s.overall}/10*",
        f"Insight:   {_score_bar(s.insight_depth)} {s.insight_depth}",
        f"Specific:  {_score_bar(s.specificity)} {s.specificity}",
        f"Timely:    {_score_bar(s.timeliness)} {s.timeliness}",
        f"LinkedIn:  {_score_bar(s.linkedin_potential)} {s.linkedin_potential}",
    ]
    return "\n".join(lines)


_KIND_HEADINGS = (
    ("research", "🔬 *Research*"),
    ("news", "📰 *News & industry*"),
    ("reference", "📚 *Reference*"),
)


def _md_safe(text: str) -> str:
    # Telegram's legacy Markdown breaks on unbalanced brackets/asterisks/underscores in titles
    return "".join(ch for ch in text if ch not in "[]*_`")


def _format_citations(r: PipelineResult) -> str:
    if not r.citations:
        return ""
    blocks = ["*Sources:*"]
    for kind, heading in _KIND_HEADINGS:
        items = [c for c in r.citations if c.kind == kind]
        if not items:
            continue
        lines = [heading]
        for c in items:
            meta = ", ".join(x for x in (_md_safe(c.source), c.date) if x)
            lines.append(f"• [{_md_safe(c.title)}]({c.url})" + (f" — {meta}" if meta else ""))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_reply(r: PipelineResult) -> str:
    if r.timed_out:
        return (
            "⏱ Timed out — the AI took too long this time.\n\n"
            "Wait 30 seconds, then resend your note."
        )

    if r.draft.startswith("⚠️"):
        return r.draft

    if r.decision == "SKIP":
        return (
            f"⏭ Not processed — {r.reason}\n\n"
            "This bot only drafts posts from skincare, formulation, or founder notes. "
            "Drop a note on those topics and I'll score and draft it."
        )

    if r.decision == "ALREADY_FORMED":
        scores = _format_scores(r)
        return f"{scores}\n\n✅ Close to post-ready.\n_{r.reason}_"

    scores  = _format_scores(r)
    sep     = "─" * 20
    sources = _format_citations(r)

    parts = [scores, sep, r.draft]
    if sources:
        parts += [sep, sources]
    return "\n\n".join(parts)


def _send(chat_id: str, text: str, reply_to: int = None, parse_mode: str = "Markdown"):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"sent ok to {chat_id} ({len(text)} chars)", file=sys.stderr)
        return "sent"
    except Exception as e:
        detail = e.read().decode()[:150] if hasattr(e, "read") else str(e)
        print(f"Telegram send error: {e} {detail}", file=sys.stderr)
        if parse_mode:
            return "plain-" + _send(chat_id, text, reply_to=reply_to, parse_mode=None)
        return f"failed({detail})"


_SEEN_UPDATES: set = set()


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if WEBHOOK_SECRET:
            token = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if token != WEBHOOK_SECRET:
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Forbidden")
                return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Respond only after processing: Vercel freezes the function as soon as the
        # response is sent, which left notes stuck on "Analysing".
        status = "OK"
        try:
            status = self._process(body) or "OK"
        finally:
            # Plain-text body is ignored by Telegram; it's a one-line status for debugging.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(status.encode())

    def _process(self, body: bytes):
        try:
            update = json.loads(body)
            update_id = update.get("update_id")
            if update_id in _SEEN_UPDATES:
                return  # Telegram retry of an update we already handled
            _SEEN_UPDATES.add(update_id)
            msg = update.get("channel_post") or update.get("message")
            if not msg:
                return

            chat_id = str(msg["chat"]["id"])
            text    = msg.get("text", "").strip()
            msg_id  = msg.get("message_id")

            if MEERA_CHAT_ID and chat_id != MEERA_CHAT_ID:
                return

            if not text or text.startswith("/"):
                if text == "/start":
                    _send(chat_id, "Drop a skincare/founder note — I'll score it and draft a LinkedIn post in Meera's voice.")
                return

            _send(chat_id, "⏳ Analysing note…", reply_to=msg_id)
            result = asyncio.run(run_pipeline(text))
            reply  = _build_reply(result)
            sent   = _send(chat_id, reply, reply_to=msg_id)
            status = (
                f"processed: decision={result.decision} score={result.scores.overall} "
                f"citations={[c.title for c in result.citations]} timed_out={result.timed_out} "
                f"reply={sent} first_line={reply.splitlines()[0][:80]!r}"
            )
            print(status, file=sys.stderr)
            return status

        except Exception as e:
            print(f"Webhook error: {e}", file=sys.stderr)
            return f"error: {e}"

    def log_message(self, format, *args):
        pass
