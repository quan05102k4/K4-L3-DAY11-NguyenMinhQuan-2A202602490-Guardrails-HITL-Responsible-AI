"""
Demo UI — chạy thử Blue pipeline (CP2–CP3) trên trình duyệt.

    python demo/server.py              # mở http://127.0.0.1:8000
    python demo/server.py --port 8080 --no-browser

Chat đi qua đúng pipeline thật trong ``src/assignment/pipeline.py``:
RateLimit → InputGuardrail → LLM Blue → OutputGuardrail → Audit + Monitoring.

Audit / metrics của demo chỉ nằm trong bộ nhớ — KHÔNG ghi đè ``outputs/*.json``.
Chỉ lắng nghe 127.0.0.1 (máy local), không expose API key ra trình duyệt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import ALLOWED_TOPICS, BLOCKED_TOPICS, blue_provider_label  # noqa: E402
from guardrails.input_guardrails import (  # noqa: E402
    InputGuardrailPlugin,
    canonicalize,
    explain_injection,
)
from guardrails.output_guardrails import content_filter, load_lab_pii_dataset  # noqa: E402
from assignment.pipeline import (  # noqa: E402
    ALLOWED_EGRESS_HOSTS,
    ATTACK_QUERIES,
    EDGE_CASES,
    SAFE_QUERIES,
    build_observability,
    build_production_plugins,
    is_egress_allowed,
    process_message,
)

INDEX_HTML = Path(__file__).with_name("index.html")
OUTPUTS = ROOT / "outputs"
LOCK = threading.Lock()  # pipeline (rate limiter, last_issues) không thread-safe → xử lý tuần tự


class DemoState:
    """Một pipeline sống suốt phiên demo (rate limiter + audit + metrics trong RAM)."""

    def __init__(self):
        self.pipeline: dict = {}
        self.reset()

    def reset(self):
        llm = self.pipeline.get("llm")  # giữ client Blue đã tạo, tránh tạo lại
        audit, monitor = build_observability()
        self.pipeline = {
            "plugins": build_production_plugins(),
            "audit": audit,
            "monitor": monitor,
        }
        if llm:
            self.pipeline["llm"] = llm


STATE = DemoState()


# ============================================================
# Giải thích từng lớp cho 1 request (trace)
# ============================================================

def _mentions(text: str, topics) -> list[str]:
    return [t for t in topics if re.search(r"\b" + re.escape(t), text)]


def _input_reason(text: str) -> str:
    limit = InputGuardrailPlugin.MAX_INPUT_CHARS
    if len(text) > limit:
        return f"Input dài {len(text):,} ký tự > giới hạn {limit:,}"
    why = explain_injection(text)
    if why:
        return f"detect_injection → BLOCK: {why}"
    canon = canonicalize(text)
    banned = _mentions(canon, BLOCKED_TOPICS)
    if banned:
        return f"topic_filter → BLOCK: chủ đề bị cấm “{banned[0]}”"
    return "topic_filter → BLOCK: không có từ khoá banking nào (ALLOWED_TOPICS)"


def _step(key: str, status: str, detail: str) -> dict:
    return {"key": key, "status": status, "detail": detail}


def build_trace(text: str, row: dict, user_id: str, call_llm: bool) -> list[dict]:
    rate_limiter, _, output_plugin = STATE.pipeline["plugins"]
    layer = row["layer"]
    used = len(rate_limiter.user_windows[user_id])
    quota = f"{used}/{rate_limiter.max_requests} request trong {rate_limiter.window_seconds}s (user “{user_id}”)"
    audit_step = _step("audit", "pass", "Đã ghi audit log + cập nhật metrics")

    if layer == "rate_limiter":
        return [
            _step("rate_limiter", "block", f"Vượt hạn mức — {quota}"),
            _step("input_guardrail", "skip", "Không chạy (đã bị chặn trước)"),
            _step("llm", "skip", "Không gọi LLM"),
            _step("output_guardrail", "skip", "Không chạy"),
            audit_step,
        ]
    steps = [_step("rate_limiter", "pass", quota)]

    if layer == "input_guardrail":
        return steps + [
            _step("input_guardrail", "block", _input_reason(text)),
            _step("llm", "skip", "Không gọi LLM — tiết kiệm chi phí, không lộ gì"),
            _step("output_guardrail", "skip", "Không chạy"),
            audit_step,
        ]
    topics = _mentions(canonicalize(text), ALLOWED_TOPICS)
    steps.append(_step(
        "input_guardrail", "pass",
        f"Không có injection · chủ đề banking: {', '.join(topics[:4])}",
    ))

    if not call_llm:
        return steps + [
            _step("llm", "skip", "Chế độ offline — không gọi LLM"),
            _step("output_guardrail", "skip", "Không có response để kiểm tra"),
            audit_step,
        ]
    if layer == "llm_error":
        return steps + [
            _step("llm", "error", "OpenRouter lỗi sau khi retry (thường là 429 của bản :free)"),
            _step("output_guardrail", "skip", "Không có response để kiểm tra"),
            audit_step,
        ]
    llm_detail = blue_provider_label()
    if layer == "model_refuse":
        llm_detail += " · model tự từ chối tiết lộ"
    steps.append(_step("llm", "pass", llm_detail))

    issues = output_plugin.last_issues
    if not issues:
        steps.append(_step("output_guardrail", "pass", "Không có PII / secret trong response"))
    else:
        status = "block" if row["blocked"] else "redact"
        verb = "Chặn secret" if row["blocked"] else "Che PII"
        steps.append(_step("output_guardrail", status, f"{verb}: {', '.join(issues)}"))
    return steps + [audit_step]


# ============================================================
# API handlers
# ============================================================

def api_chat(body: dict) -> dict:
    message = str(body.get("message", ""))
    user_id = (str(body.get("user_id") or "demo-user")).strip()[:40] or "demo-user"
    call_llm = bool(body.get("call_llm", True))
    with LOCK:
        row = asyncio.run(
            process_message(STATE.pipeline, message, user_id=user_id, call_llm=call_llm)
        )
        entry = STATE.pipeline["audit"].logs[-1]  # audit giữ reply đầy đủ (đã qua output guardrail)
        return {
            "reply": entry["output"],
            "blocked": row["blocked"],
            "layer": row["layer"],
            "latency_ms": entry["latency_ms"],
            "trace": build_trace(message, row, user_id, call_llm),
            "state": api_state(),
        }


def api_state() -> dict:
    p = STATE.pipeline
    monitor = p["monitor"]
    monitor.check_metrics()
    logs = p["audit"].logs[-12:]
    return {
        "metrics": monitor.snapshot(),
        "audit": [
            {
                "time": e["timestamp"][11:19],
                "user_id": e["user_id"],
                "input": (e["input"] or "")[:80],
                "blocked": e["blocked"],
                "layer": e["layer"],
                "latency_ms": e["latency_ms"],
            }
            for e in reversed(logs)
        ],
    }


def api_egress(body: dict) -> dict:
    destination = str(body.get("destination", ""))
    payload = str(body.get("payload", ""))
    checks = []
    try:
        url = urlparse(destination.strip())
        port = url.port
    except ValueError:
        url, port = None, None
        checks.append({"label": "URL hợp lệ", "ok": False, "detail": "Không parse được URL / port"})
    if url is not None:
        checks.append({"label": "Giao thức HTTPS", "ok": url.scheme == "https",
                       "detail": f"scheme = {url.scheme or '(trống)'}"})
        checks.append({"label": "Host trong allowlist", "ok": url.hostname in ALLOWED_EGRESS_HOSTS,
                       "detail": f"host = {url.hostname or '(trống)'} · cho phép: {', '.join(sorted(ALLOWED_EGRESS_HOSTS))}"})
        clean = not (url.username or url.password) and port in (None, 443)
        checks.append({"label": "Không userinfo, port 443", "ok": clean,
                       "detail": f"userinfo = {'có' if url.username else 'không'} · port = {port or 'mặc định'}"})
    filtered = content_filter(payload)
    checks.append({"label": "Payload không chứa PII / secret", "ok": filtered["safe"],
                   "detail": ", ".join(filtered["issues"]) or "sạch"})
    return {"allowed": is_egress_allowed(destination, payload), "checks": checks}


def _load(name: str):
    path = OUTPUTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def api_summary() -> dict:
    dataset = load_lab_pii_dataset()["pii_cases"]
    pii_ok = 0
    for case in dataset:
        r = content_filter(case["input_text"])
        found = {i.split(":")[0] for i in r["issues"]}
        pii_ok += r["safe"] == case["expect_safe"] and found == set(case["expect_issue_types"])

    results, metrics, attacks = _load("results.json"), _load("metrics.json"), _load("attack_results.json")
    cp4 = None
    if attacks:
        red, adv = attacks.get("unsafe_attacks", []), attacks.get("guards_attacks", [])
        by_id = {g["id"]: g for g in adv}
        cp4 = {
            "provider": attacks.get("llm_provider"),
            "model": attacks.get("llm_model"),
            "red": {"total": len(red), "leaked": sum(r["leaked"] for r in red)},
            "advance": {"total": len(adv), "leaked": sum(r["leaked"] for r in adv)},
            "rows": [
                {
                    "id": r["id"],
                    "category": r["category"],
                    "red": r["layer"],
                    "advance": (by_id.get(r["id"]) or {}).get("layer"),
                }
                for r in red
            ],
        }
    return {
        "cp2": {"pii_ok": pii_ok, "pii_total": len(dataset)},
        "cp3": None if not results else {
            "summary": results.get("summary"),
            "rate_limit": results.get("rate_limit"),
            "plugin_order": results.get("plugin_order"),
            "blue_model": results.get("blue_model"),
            "egress": results.get("egress_checks"),
        },
        "metrics": metrics,
        "cp4": cp4,
    }


def api_samples() -> dict:
    def label(text: str) -> str:
        if len(text) > 200:
            return f"“{text[:1]}” × {len(text):,} ký tự"
        return text if text else "(chuỗi rỗng)"

    group = lambda items: [{"label": label(t), "text": t} for t in items]  # noqa: E731
    return {"safe": group(SAFE_QUERIES), "attack": group(ATTACK_QUERIES), "edge": group(EDGE_CASES)}


# ============================================================
# HTTP server (stdlib — không thêm dependency)
# ============================================================

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200):
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            with LOCK:
                self._json(api_state())
        elif path == "/api/summary":
            self._json(api_summary())
        elif path == "/api/samples":
            self._json(api_samples())
        elif path == "/api/info":
            self._json({"blue_model": blue_provider_label()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if path == "/api/chat":
                self._json(api_chat(body))
            elif path == "/api/egress":
                self._json(api_egress(body))
            elif path == "/api/reset":
                with LOCK:
                    STATE.reset()
                    self._json(api_state())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # trả lỗi cho UI thay vì làm rơi kết nối
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):  # gọn terminal: chỉ log API POST
        if self.command == "POST":
            sys.stderr.write(f"  {self.command} {self.path} → {args[1] if len(args) > 1 else ''}\n")


def main():
    parser = argparse.ArgumentParser(description="Demo UI cho Blue pipeline (Lab Day 11)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Demo UI → {url}   (Ctrl+C để dừng)")
    print(f"Blue: {blue_provider_label()} · outputs/*.json KHÔNG bị ghi đè")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
