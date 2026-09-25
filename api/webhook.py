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


def _format_citations(r: PipelineResult) -> str:
    if not r.citations:
        return ""
    lines = ["📰 *Sources cited:*"]
    for c in r.citations:
        line = f"• [{c.title}]({c.url})"
        if c.source:
            line += f" — {c.source}"
        lines.append(line)
    return "\n".join(lines)


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
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Telegram send error: {e}", file=sys.stderr)


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

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

        try:
            update = json.loads(body)
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
            _send(chat_id, reply, reply_to=msg_id)

        except Exception as e:
            print(f"Webhook error: {e}", file=sys.stderr)

    def log_message(self, format, *args):
        pass
