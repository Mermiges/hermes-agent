"""Family Ant harness bridge for gateway slash commands."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from gateway.family_ant_bridge_format import one_line, payload_questions, summarize_payload


DEFAULT_REPO_ROOT = Path("/mnt/hdd/family-ant_harness_refactor")
DEFAULT_HERMES_HOME = Path("/home/mermiges/.hermes")
DEFAULT_PYTHON = Path("/usr/bin/python3")
ORCH_STATE_ROOT = Path("runs/hermes-orchnl")
LOCAL_MODE = "local_first"
DRY_TIMEOUT_SECONDS = 900
RUN_TIMEOUT_SECONDS = 7200
STATUS_TIMEOUT_SECONDS = 45
STATUS_READINESS_TIMEOUT_SECONDS = 4

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"),
)
_FRONTIER_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "NOUS_API_KEY",
        "KIMI_API_KEY",
        "GLM_API_KEY",
    }
)
_BRAIN_ALIASES = {
    "codex": "codex_cli",
    "codex_cli": "codex_cli",
    "qwopus_boner": "qwopus_boner",
    "qwopus_fart": "qwopus_fart",
}
_MANUAL_RUN_WORKFLOWS = frozenset(
    {
        "finalization_export_review_gate",
        "gmail_ingest",
        "service_email",
        "trial_package_delivery",
    }
)


@dataclass
class HarnessSession:
    state_dir: str
    status: str
    has_questions: bool
    pending_request: str = ""


@dataclass(frozen=True)
class MatterTarget:
    matter_id: str
    matter_path: Path
    request: str


class BridgeUserQuestion(Exception):
    """A user-answerable intake question, not an infrastructure failure."""


class HarnessGatewayBridge:
    """Run Family Ant Hermes orchestration from gateway slash commands."""

    def __init__(
        self,
        *,
        repo_root: Path = DEFAULT_REPO_ROOT,
        hermes_home: Path = DEFAULT_HERMES_HOME,
        python_executable: Path = DEFAULT_PYTHON,
    ) -> None:
        self.repo_root = repo_root.expanduser().resolve(strict=False)
        self.hermes_home = hermes_home.expanduser().resolve(strict=False)
        self.python_executable = python_executable
        self.sessions: dict[str, HarnessSession] = {}
        self._chat_memory_impl = None

    async def plan(self, session_key: str, request: str) -> str:
        request = request.strip()
        self._append_chat(session_key, role="user", text=request, event="plan")
        self._log_event("telegram_inbound", session_key=session_key, text=request, status="plan")
        if not request:
            return "Usage: /harness <request with client shorthand>"
        if _normalize_text(request) in _PLAIN_STOP:
            self.sessions.pop(session_key, None)
            self._update_memory(
                session_key,
                state_dir="",
                status="stopped",
                has_questions=False,
                pending_request="",
            )
            reply = "Stopped. I will wait for your next instruction."
            self._append_chat(session_key, role="assistant", text=reply, event="stopped")
            return reply
        followup = self._completed_followup(session_key, request)
        if followup:
            return followup
        try:
            target = self._resolve_matter(request, session_key=session_key)
        except BridgeUserQuestion as exc:
            self.sessions[session_key] = HarnessSession(
                state_dir="",
                status="intake_question",
                has_questions=True,
                pending_request=request,
            )
            self._update_memory(
                session_key,
                status="intake_question",
                has_questions=True,
                pending_request=request,
            )
            self._log_event(
                "telegram_plan_question",
                session_key=session_key,
                text=request,
                status="intake_question",
                error=str(exc),
            )
            reply = f"Questions:\n- matter_id: {one_line(str(exc), 500)}\nNext: reply normally with the client/matter."
            self._append_chat(session_key, role="assistant", text=reply, event="intake_question")
            return reply
        routed_request = _request_with_memory_context(target.request, self._chat_context(session_key))
        self._update_memory(
            session_key,
            matter_id=target.matter_id,
            matter_path=str(target.matter_path),
        )
        try:
            payload = await self._run_orchestrate(
                [
                    "orchestrate",
                    routed_request,
                    "--dry-run",
                    "--matter-id",
                    target.matter_id,
                    "--matter-path",
                    str(target.matter_path),
                ],
                timeout=DRY_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            self._log_event(
                "telegram_plan_error",
                session_key=session_key,
                text=request,
                status=type(exc).__name__,
                error=str(exc),
            )
            raise
        self._remember(session_key, payload, request=target.request)
        self._log_event(
            "telegram_plan_result",
            session_key=session_key,
            text=target.request,
            payload=payload,
            status=str(payload.get("status") or "unknown"),
            state_dir=str(payload.get("state_dir") or ""),
        )
        if _should_auto_run_payload(payload):
            run_payload = await self._run_orchestrate(
                ["orchestrate", "--resume", str(payload.get("state_dir") or "")],
                timeout=RUN_TIMEOUT_SECONDS,
            )
            self._remember(session_key, run_payload, request=target.request)
            self._log_event(
                "telegram_auto_run_result",
                session_key=session_key,
                text=target.request,
                payload=run_payload,
                status=str(run_payload.get("status") or "unknown"),
                state_dir=str(run_payload.get("state_dir") or ""),
            )
            reply = self._associate_response(
                session_key,
                request=target.request,
                run_payload=run_payload,
                plan_payload=payload,
            )
            reply = _auto_run_reply(payload, run_payload, rendered=reply)
            self._append_chat(session_key, role="assistant", text=reply, event="auto_run_result")
            return reply
        lines = [summarize_payload(payload)]
        if payload_questions(payload):
            lines.append("I need the missing detail before I can proceed.")
        elif str(payload.get("status") or "") in {"failed", "invalid_plan"}:
            lines.append("I will not run this automatically; the diagnostic blocker needs to be fixed or the request retried.")
        elif _manual_run_workflow_labels(payload):
            lines.append(
                "I have this saved, but Telegram will not auto-run service, delivery, "
                "court-facing finalization/export, or external account-sync workflows. "
                "Review the documents, recipients, and external-action boundary before running it manually."
            )
        else:
            lines.append("I can run this now unless you say stop.")
        reply = "\n".join(line for line in lines if line).strip()
        self._append_chat(session_key, role="assistant", text=reply, event="plan_result")
        return reply

    async def go(self, session_key: str) -> str:
        self._log_event("telegram_go", session_key=session_key, status="requested")
        session = self._session(session_key)
        if session is None or not session.state_dir:
            self._log_event("telegram_go_rejected", session_key=session_key, status="no_saved_plan")
            return "I do not have a saved task to run. Please send the task you want handled."
        if session.has_questions:
            self._log_event("telegram_go_rejected", session_key=session_key, status="pending_questions")
            return "I need the missing detail before I can proceed."
        try:
            payload = await self._run_orchestrate(
                ["orchestrate", "--resume", session.state_dir],
                timeout=RUN_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            self._log_event(
                "telegram_run_error",
                session_key=session_key,
                status=type(exc).__name__,
                state_dir=session.state_dir,
                error=str(exc),
            )
            raise
        self._remember(session_key, payload, request=session.pending_request)
        self._log_event(
            "telegram_run_result",
            session_key=session_key,
            payload=payload,
            status=str(payload.get("status") or "unknown"),
            state_dir=str(payload.get("state_dir") or session.state_dir),
        )
        reply = self._associate_response(
            session_key,
            request=session.pending_request or "",
            run_payload=payload,
        )
        self._append_chat(session_key, role="assistant", text=reply, event="run_result")
        return reply

    async def answer(self, session_key: str, answer_text: str) -> str:
        answer_text = answer_text.strip()
        self._append_chat(session_key, role="user", text=answer_text, event="answer")
        self._log_event("telegram_inbound", session_key=session_key, text=answer_text, status="answer")
        if not answer_text:
            return "I need the missing detail before I can proceed."
        if _normalize_text(answer_text) in _PLAIN_STOP:
            self.sessions.pop(session_key, None)
            self._update_memory(
                session_key,
                state_dir="",
                status="stopped",
                has_questions=False,
                pending_request="",
            )
            reply = "Stopped. I will wait for your next instruction."
            self._append_chat(session_key, role="assistant", text=reply, event="stopped")
            return reply
        followup = self._completed_followup(session_key, answer_text)
        if followup:
            return followup
        session = self._session(session_key)
        if session is None or not session.state_dir:
            if session is not None and session.pending_request:
                return await self.plan(
                    session_key,
                    f"{answer_text}: {session.pending_request}",
                )
            self._log_event("telegram_answer_rejected", session_key=session_key, text=answer_text, status="no_saved_plan")
            return "I do not have a saved task to run. Please send the task you want handled."
        try:
            payload = await self._run_orchestrate(
                [
                    "orchestrate",
                    "--resume",
                    session.state_dir,
                    "--answer",
                    answer_text,
                    "--dry-run",
                ],
                timeout=DRY_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            self._log_event(
                "telegram_answer_error",
                session_key=session_key,
                text=answer_text,
                status=type(exc).__name__,
                state_dir=session.state_dir,
                error=str(exc),
            )
            raise
        self._remember(session_key, payload, request=session.pending_request or answer_text)
        self._log_event(
            "telegram_answer_result",
            session_key=session_key,
            text=answer_text,
            payload=payload,
            status=str(payload.get("status") or "unknown"),
            state_dir=str(payload.get("state_dir") or session.state_dir),
        )
        if _should_auto_run_payload(payload):
            run_payload = await self._run_orchestrate(
                ["orchestrate", "--resume", str(payload.get("state_dir") or session.state_dir)],
                timeout=RUN_TIMEOUT_SECONDS,
            )
            self._remember(session_key, run_payload, request=session.pending_request or answer_text)
            self._log_event(
                "telegram_auto_run_result",
                session_key=session_key,
                text=answer_text,
                payload=run_payload,
                status=str(run_payload.get("status") or "unknown"),
                state_dir=str(run_payload.get("state_dir") or ""),
            )
            reply = self._associate_response(
                session_key,
                request=answer_text,
                run_payload=run_payload,
                plan_payload=payload,
            )
            reply = _auto_run_reply(payload, run_payload, rendered=reply)
            self._append_chat(session_key, role="assistant", text=reply, event="auto_run_result")
            return reply
        lines = [summarize_payload(payload)]
        if payload_questions(payload):
            lines.append("I need the missing detail before I can proceed.")
        elif str(payload.get("status") or "") in {"failed", "invalid_plan"}:
            lines.append("I will not run this automatically; the diagnostic blocker needs to be fixed or the request retried.")
        elif _manual_run_workflow_labels(payload):
            lines.append(
                "I have this saved, but Telegram will not auto-run service, delivery, "
                "court-facing finalization/export, or external account-sync workflows. "
                "Review the documents, recipients, and external-action boundary before running it manually."
            )
        else:
            lines.append("I can run this now unless you say stop.")
        reply = "\n".join(line for line in lines if line).strip()
        self._append_chat(session_key, role="assistant", text=reply, event="answer_result")
        return reply

    def _completed_followup(self, session_key: str, text: str) -> str:
        if not _looks_like_completed_followup(text):
            return ""
        payload = self._latest_completed_payload(session_key)
        if not payload:
            return ""
        reply = self._associate_response(
            session_key,
            request=text,
            run_payload=payload,
        )
        if not reply:
            return ""
        self._remember(session_key, payload, request=text)
        self._log_event(
            "telegram_followup_result",
            session_key=session_key,
            text=text,
            payload=payload,
            status="completed_followup",
            state_dir=str(payload.get("state_dir") or ""),
        )
        self._append_chat(session_key, role="assistant", text=reply, event="followup_result")
        return reply

    def _associate_response(
        self,
        session_key: str,
        *,
        request: str,
        run_payload: Mapping[str, Any],
        plan_payload: Mapping[str, Any] | None = None,
    ) -> str:
        self._ensure_harness_imports()
        from hermes.associate_response import render_associate_response

        return render_associate_response(
            request=request,
            run_payload=run_payload,
            plan_payload=plan_payload,
            repo_root=self.repo_root,
            hermes_home=self.hermes_home,
            chat_context=self._chat_context(session_key),
        )

    def _latest_completed_payload(self, session_key: str) -> dict[str, Any] | None:
        candidates: list[str] = []
        session = self._session(session_key)
        if session is not None and session.state_dir:
            candidates.append(session.state_dir)
        try:
            raw = self._chat_memory().session(session_key)
        except Exception:
            raw = {}
        messages = raw.get("messages") if isinstance(raw, Mapping) else None
        if isinstance(messages, list):
            for item in reversed(messages):
                if not isinstance(item, Mapping):
                    continue
                text = str(item.get("text") or "")
                if "Status: completed" not in text:
                    continue
                candidates.extend(_state_names_from_text(text))
        payload_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            payload = self._load_state_payload(candidate)
            if _is_completed_payload_with_answer(payload):
                payload_candidates.append(payload)
        payload_candidates.extend(
            self._completed_payloads_from_runs(raw if isinstance(raw, Mapping) else {})
        )
        return _best_completed_followup_payload(payload_candidates)

    def _completed_payloads_from_runs(self, memory: Mapping[str, Any]) -> list[dict[str, Any]]:
        root = self.repo_root / ORCH_STATE_ROOT
        if not root.exists():
            return []
        remembered_matter_id = str(memory.get("matter_id") or "").strip()
        remembered_matter_path = str(memory.get("matter_path") or "").strip()
        rows: list[tuple[float, Path]] = []
        for state_file in root.glob("*/state.json"):
            try:
                rows.append((state_file.stat().st_mtime, state_file))
            except OSError:
                continue
        payloads: list[dict[str, Any]] = []
        for _mtime, state_file in sorted(rows, reverse=True)[:20]:
            payload = self._load_state_payload(str(state_file))
            if not _is_completed_payload_with_answer(payload):
                continue
            plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
            matter_id = str(plan.get("matter_id") or "").strip()
            matter_path = str(plan.get("matter_path") or "").strip()
            if remembered_matter_id and matter_id and remembered_matter_id != matter_id:
                continue
            if remembered_matter_path and matter_path and remembered_matter_path != matter_path:
                continue
            payloads.append(payload)
        return payloads

    def _load_state_payload(self, raw_state_dir: str) -> dict[str, Any] | None:
        raw_state_dir = str(raw_state_dir or "").strip()
        if not raw_state_dir:
            return None
        raw = Path(raw_state_dir).expanduser()
        candidates: list[Path] = []
        if raw.name == "state.json":
            candidates.append(raw)
        else:
            candidates.append(raw / "state.json")
            candidates.append(self.repo_root / ORCH_STATE_ROOT / raw.name / "state.json")
        for state_file in candidates:
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def brain(self, raw_name: str) -> str:
        self._ensure_harness_imports()
        from hermes.orchnl_brain import load_router_brain, router_brain_names, save_router_brain

        requested = raw_name.strip().lower()
        if requested:
            canonical = _BRAIN_ALIASES.get(requested)
            if canonical is None:
                allowed = "codex, qwopus_boner, qwopus_fart"
                return f"Unknown brain {requested!r}. Allowed: {allowed}."
            save_router_brain(canonical, hermes_home=self.hermes_home)
        current = load_router_brain(hermes_home=self.hermes_home).name
        allowed = ", ".join(router_brain_names())
        return (
            f"Current Family Ant router brain: {current}\n"
            f"Allowed: {allowed}\n"
            "This changes the harness orchestrator planning brain only. "
            f"Gateway chat model remains separate: {_gateway_model_line()}."
        )

    async def status(self) -> str:
        brain_line = self.brain("").splitlines()[0]
        fleet = await self._fleet_status()
        states = self._last_states()
        lines = [brain_line, fleet]
        if states:
            lines.append("Last orchestration states:")
            lines.extend(f"- {name}: {status}" for name, status in states)
        else:
            lines.append("Last orchestration states: none")
        return "\n".join(lines)

    def failure_reply(self, session_key: str, request: str, exc: Exception) -> str:
        """Recover a structured diagnostic when an older bridge path raises raw text."""

        detail = f"{type(exc).__name__}: {safe_error_line(str(exc))}"
        payload = self._synthesize_failed_orchestration_payload(
            args=["orchestrate", request, "--dry-run"],
            detail=detail,
            reason="Gateway caught a Family Ant bridge exception before a structured payload was returned.",
        )
        self._remember(session_key, payload, request=request)
        self._log_event(
            "telegram_bridge_failure_synthesized",
            session_key=session_key,
            text=request,
            payload=payload,
            status=str(payload.get("status") or "failed"),
            state_dir=str(payload.get("state_dir") or ""),
            error=detail,
        )
        reply = "\n".join(
            [
                summarize_payload(payload),
                "I will not run this automatically; the diagnostic blocker needs to be fixed or the request retried.",
            ]
        ).strip()
        self._append_chat(session_key, role="assistant", text=reply, event="failure_synthesized")
        return reply

    def _remember(self, session_key: str, payload: Mapping[str, Any], *, request: str | None = None) -> None:
        state_dir = str(payload.get("state_dir") or "").strip()
        if not state_dir:
            return
        plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
        matter_id = str(plan.get("matter_id") or "").strip()
        matter_path = str(plan.get("matter_path") or "").strip()
        has_questions = bool(payload_questions(payload))
        existing = self._session(session_key)
        remembered: Mapping[str, Any] = {}
        if not matter_id or not matter_path:
            try:
                raw_memory = self._chat_memory().session(session_key)
            except Exception:
                raw_memory = {}
            if isinstance(raw_memory, Mapping):
                remembered = raw_memory
        if not matter_id:
            matter_id = str(remembered.get("matter_id") or "").strip()
        if not matter_path:
            matter_path = str(remembered.get("matter_path") or "").strip()
        pending_request = (request or (existing.pending_request if existing is not None else "") or "").strip()
        self.sessions[session_key] = HarnessSession(
            state_dir=state_dir,
            status=str(payload.get("status") or "unknown"),
            has_questions=has_questions,
            pending_request=pending_request,
        )
        self._update_memory(
            session_key,
            state_dir=state_dir,
            status=str(payload.get("status") or "unknown"),
            has_questions=has_questions,
            pending_request=pending_request,
            matter_id=matter_id,
            matter_path=matter_path,
        )

    def _session(self, session_key: str) -> HarnessSession | None:
        session = self.sessions.get(session_key)
        if session is not None:
            return session
        try:
            raw = self._chat_memory().session(session_key)
        except Exception:
            return None
        state_dir = str(raw.get("state_dir") or "").strip()
        pending_request = str(raw.get("pending_request") or "").strip()
        has_questions = bool(raw.get("has_questions"))
        if not (state_dir or pending_request or has_questions):
            return None
        session = HarnessSession(
            state_dir=state_dir,
            status=str(raw.get("status") or "unknown"),
            has_questions=has_questions,
            pending_request=pending_request,
        )
        self.sessions[session_key] = session
        return session

    def has_saved_plan(self, session_key: str) -> bool:
        session = self._session(session_key)
        return bool(session and session.state_dir)

    def has_pending_questions(self, session_key: str) -> bool:
        session = self._session(session_key)
        return bool(session and session.has_questions)

    def _chat_memory(self):
        if self._chat_memory_impl is None:
            self._ensure_harness_imports()
            from hermes.chat_memory import HermesChatMemory

            self._chat_memory_impl = HermesChatMemory(hermes_home=self.hermes_home)
        return self._chat_memory_impl

    def _chat_context(self, session_key: str) -> str:
        try:
            return self._chat_memory().recent_context(session_key)
        except Exception:
            return ""

    def _append_chat(self, session_key: str, *, role: str, text: str, event: str) -> None:
        try:
            self._chat_memory().append_message(session_key, role=role, text=text, event=event)
        except Exception:
            return

    def _update_memory(self, session_key: str, **kwargs: Any) -> None:
        try:
            self._chat_memory().update_session(session_key, **kwargs)
        except Exception:
            return

    def _log_event(
        self,
        event: str,
        *,
        session_key: str,
        text: str | None = None,
        payload: Mapping[str, Any] | None = None,
        status: str | None = None,
        state_dir: str | None = None,
        error: str | None = None,
    ) -> None:
        try:
            self._ensure_harness_imports()
            from hermes.prompt_ledger import log_hermes_event

            log_hermes_event(
                event,
                source="hermes_gateway_bridge",
                text=text,
                session_key=session_key,
                payload=payload,
                status=status,
                state_dir=state_dir,
                error=error,
            )
        except Exception:
            return

    async def _run_orchestrate(self, args: list[str], *, timeout: int) -> dict[str, Any]:
        command = [
            str(self.python_executable),
            "-m",
            "tools.hermes",
            *args,
            "--repo-root",
            str(self.repo_root),
            "--hermes-home",
            str(self.hermes_home),
        ]
        completed = await self._run(command, timeout=timeout, ok_returncodes=(0, 1, 2))
        try:
            payload = json.loads(completed[0])
        except json.JSONDecodeError as exc:
            combined = "\n".join(part for part in completed if str(part or "").strip())
            if _looks_like_harness_summary_failure(combined):
                return self._synthesize_failed_orchestration_payload(
                    args=args,
                    detail=combined,
                    reason=f"Harness returned non-JSON summary output: {exc}",
                )
            err = safe_error_line(combined)
            raise RuntimeError(f"Harness returned non-JSON output: {exc}; {err}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Harness JSON payload was not an object.")
        return payload

    def _synthesize_failed_orchestration_payload(
        self,
        *,
        args: list[str],
        detail: str,
        reason: str,
    ) -> dict[str, Any]:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        state_dir = self.repo_root / ORCH_STATE_ROOT / f"gateway-failure-{stamp}"
        request = _orchestrate_request_from_args(args)
        matter_id = _flag_value(args, "--matter-id")
        matter_path = _flag_value(args, "--matter-path") or str(self.repo_root)
        error = f"gateway_orchestrate_failed:{safe_error_line(reason or detail)}"
        plan = {
            "version": "orchnl.v1",
            "request": request,
            "matter_id": matter_id,
            "matter_path": matter_path,
            "summary": "Hermes failed before it could build an executable plan.",
            "steps": [],
            "questions": [],
        }
        payload: dict[str, Any] = {
            "status": "failed",
            "dry_run": "--dry-run" in args,
            "state_dir": str(state_dir),
            "plan": plan,
            "validation": {
                "ok": False,
                "executable": False,
                "errors": [error],
                "questions": [],
            },
            "steps": [],
            "memory_diffs": [],
            "answers": [],
            "error": safe_error_line(reason or detail),
            "router": {"planning_error": error},
        }
        self._write_gateway_failure_state(state_dir, payload)
        advice = self._codex_failure_advice(payload, state_dir=state_dir)
        if advice is not None:
            payload["failure_advice"] = advice
            self._write_gateway_failure_state(state_dir, payload)
        return payload

    def _write_gateway_failure_state(self, state_dir: Path, payload: Mapping[str, Any]) -> None:
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
            (state_dir / "plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            (state_dir / "state.json").write_text(
                json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        except Exception:
            return

    def _codex_failure_advice(self, payload: Mapping[str, Any], *, state_dir: Path) -> dict[str, Any] | None:
        old_enabled = os.environ.get("HERMES_CODEX_FAILURE_ADVISOR")
        old_timeout = os.environ.get("HERMES_CODEX_FAILURE_ADVISOR_TIMEOUT")
        os.environ["HERMES_CODEX_FAILURE_ADVISOR"] = "1"
        os.environ.setdefault("HERMES_CODEX_FAILURE_ADVISOR_TIMEOUT", "45")
        try:
            self._ensure_harness_imports()
            from hermes.failure_advisor import codex_failure_advice

            return codex_failure_advice(payload=payload, repo_root=self.repo_root, state_dir=state_dir)
        except Exception as exc:  # noqa: BLE001 - gateway failure synthesis must not mask the original failure.
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {safe_error_line(str(exc))}"}
        finally:
            if old_enabled is None:
                os.environ.pop("HERMES_CODEX_FAILURE_ADVISOR", None)
            else:
                os.environ["HERMES_CODEX_FAILURE_ADVISOR"] = old_enabled
            if old_timeout is None:
                os.environ.pop("HERMES_CODEX_FAILURE_ADVISOR_TIMEOUT", None)
            else:
                os.environ["HERMES_CODEX_FAILURE_ADVISOR_TIMEOUT"] = old_timeout

    async def _fleet_status(self) -> str:
        command = [
            str(self.python_executable),
            "tools/stack_status.py",
            "--format",
            "json",
            "--readiness-probe",
            "--readiness-timeout",
            str(STATUS_READINESS_TIMEOUT_SECONDS),
        ]
        try:
            stdout, stderr = await self._run(command, timeout=STATUS_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - status should report partial failure.
            return f"Fleet health: unavailable ({safe_error_line(str(exc))})"
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return f"Fleet health: unavailable ({safe_error_line(stderr or stdout)})"
        rows = payload.get("consumers") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return "Fleet health: unavailable"
        checked = [row for row in rows if isinstance(row, dict) and row.get("health_url")]
        up = sum(1 for row in checked if row.get("health") is True)
        down = [str(row.get("name") or "?") for row in checked if row.get("health") is False]
        not_ready = [
            str(row.get("name") or "?")
            for row in checked
            if isinstance(row.get("readiness"), Mapping) and row["readiness"].get("ok") is False
        ]
        text = f"Fleet health: {up} up, {len(down)} down"
        if down:
            text += " (" + ", ".join(down[:6]) + (", ..." if len(down) > 6 else "") + ")"
        if not_ready:
            text += "; readiness failures: " + ", ".join(not_ready[:6]) + (", ..." if len(not_ready) > 6 else "")
        return text

    async def _run(
        self,
        command: list[str],
        *,
        timeout: int,
        ok_returncodes: tuple[int, ...] = (0, 2),
    ) -> tuple[str, str]:
        env = self._subprocess_env()
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.repo_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(f"Harness command timed out after {timeout}s") from exc
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if proc.returncode not in ok_returncodes:
            raise RuntimeError(safe_error_line(stderr or stdout))
        return stdout, stderr

    def _subprocess_env(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in _FRONTIER_ENV_KEYS}
        env["FAMILY_ANT_EXECUTION_MODE"] = LOCAL_MODE
        env["FAMILY_ANT_HERMES_HOME"] = str(self.hermes_home)
        env["HERMES_CODEX_FAILURE_ADVISOR"] = "1"
        env.setdefault("HERMES_CODEX_FAILURE_ADVISOR_TIMEOUT", "45")
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = _prepend_pythonpath(self.repo_root, env.get("PYTHONPATH"))
        return env

    def _resolve_matter(self, request: str, *, session_key: str) -> MatterTarget:
        self._ensure_harness_imports()
        from hermes.orchnl_intake import (
            ClientResolutionAmbiguous,
            IntakeResolutionError,
            clean_client_hint,
            resolve_client_reference,
            resolve_matter_intake,
        )

        try:
            intake = resolve_matter_intake(
                request=request,
                matter_id="",
                matter_path=None,
                repo_root=self.repo_root,
            )
            if intake.source != "repo_fallback":
                return MatterTarget(_slug(intake.matter_id), intake.matter_path, intake.request)
        except ClientResolutionAmbiguous as exc:
            raise BridgeUserQuestion(_ambiguous_message(exc.reference, exc.candidates)) from exc
        except (FileNotFoundError, IntakeResolutionError):
            pass

        first_line = request.strip().splitlines()[0] if request.strip() else ""
        prefix = first_line.split(":", 1)[0] if ":" in first_line[:120] else first_line
        cleaned = clean_client_hint(prefix)
        words = cleaned.split()
        for count in range(min(4, len(words)), 0, -1):
            hint = " ".join(words[:count])
            try:
                path = resolve_client_reference(hint)
                return MatterTarget(_slug(path.name), path, request)
            except ClientResolutionAmbiguous as exc:
                raise BridgeUserQuestion(_ambiguous_message(exc.reference, exc.candidates)) from exc
            except FileNotFoundError:
                continue
        remembered = self._remembered_matter(session_key)
        if remembered is not None:
            matter_id, matter_path = remembered
            return MatterTarget(matter_id, matter_path, request)
        raise BridgeUserQuestion(
            "Which client folder should Hermes use? Start the request with an exact or unique client shorthand."
        )

    def _remembered_matter(self, session_key: str) -> tuple[str, Path] | None:
        try:
            raw = self._chat_memory().session(session_key)
        except Exception:
            return None
        matter_path = str(raw.get("matter_path") or "").strip()
        if not matter_path:
            return None
        path = Path(matter_path).expanduser().resolve(strict=False)
        if not path.exists():
            return None
        try:
            self._ensure_harness_imports()
            from hermes.orchnl_intake import assert_not_closed_client_path

            assert_not_closed_client_path(path)
        except Exception:
            return None
        matter_id = _slug(str(raw.get("matter_id") or path.name))
        return matter_id, path

    def _last_states(self) -> list[tuple[str, str]]:
        root = self.repo_root / ORCH_STATE_ROOT
        rows: list[tuple[float, str, str]] = []
        if not root.exists():
            return []
        for state_file in root.glob("*/state.json"):
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
                status = str(payload.get("status") or "unknown")
                rows.append((state_file.stat().st_mtime, state_file.parent.name, status))
            except (OSError, json.JSONDecodeError):
                continue
        rows.sort(reverse=True)
        return [(name, status) for _, name, status in rows[:3]]

    def _ensure_harness_imports(self) -> None:
        repo = str(self.repo_root)
        if repo not in sys.path:
            sys.path.append(repo)


def _raw_case_search_answer(steps: Any) -> str:
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if step.get("kind") != "case_search" and step.get("tool") != "case_search":
            continue
        payload = step.get("payload") if isinstance(step.get("payload"), Mapping) else {}
        answer = str(payload.get("answer") or "").strip()
        if answer:
            return answer
    return ""


def _is_completed_payload_with_answer(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if str(payload.get("status") or "") != "completed":
        return False
    return bool(_raw_case_search_answer(payload.get("steps")))


def _auto_run_reply(
    plan_payload: Mapping[str, Any],
    run_payload: Mapping[str, Any],
    *,
    rendered: str = "",
) -> str:
    lines = [_auto_run_lead(plan_payload, run_payload)]
    rendered = rendered or _completed_followup_summary("", run_payload)
    lines.append(rendered or summarize_payload(run_payload))
    return "\n".join(line for line in lines if line).strip()[:3900]


def _auto_run_lead(plan_payload: Mapping[str, Any], run_payload: Mapping[str, Any]) -> str:
    status = str(run_payload.get("status") or "").casefold()
    action = _plain_action_summary(plan_payload)
    if status in {"completed", "needs_review"}:
        return "I will " + action + "."
    if status in {"failed", "invalid_plan"}:
        return "I tried to " + action + ", but Hermes failed before producing reviewable work."
    return ""


def _plain_action_summary(payload: Mapping[str, Any]) -> str:
    plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
    labels = _workflow_labels(plan.get("steps") if isinstance(plan.get("steps"), list) else [])
    if labels == ["case_search"]:
        return "search the local case file and return a source-supported analysis"
    if labels:
        return "run " + ", ".join(labels)
    summary = one_line(str(plan.get("summary") or ""), 220)
    return summary or "run the requested local Hermes task"


def _should_auto_run_payload(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("status") or "") != "dry_run":
        return False
    if not str(payload.get("state_dir") or "").strip():
        return False
    if payload_questions(payload):
        return False
    plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    labels = _workflow_labels(steps)
    if _manual_run_labels(labels):
        return False
    return bool(labels)


def _manual_run_workflow_labels(payload: Mapping[str, Any]) -> list[str]:
    plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    return _manual_run_labels(_workflow_labels(steps))


def _manual_run_labels(labels: list[str]) -> list[str]:
    manual: list[str] = []
    for label in labels:
        normalized = str(label or "").strip().casefold().replace("-", "_")
        if normalized in _MANUAL_RUN_WORKFLOWS:
            manual.append(label)
    return manual


def _workflow_labels(steps: list[Any]) -> list[str]:
    labels: list[str] = []
    for item in steps:
        if not isinstance(item, Mapping):
            continue
        if item.get("kind") == "case_search":
            label = "case_search"
        else:
            label = str(item.get("workflow") or item.get("tool") or item.get("kind") or "").strip()
        if label:
            labels.append(label)
    return labels


def _best_completed_followup_payload(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    unique: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        key = str(payload.get("state_dir") or id(payload))
        unique[key] = payload
    ranked = sorted(
        unique.values(),
        key=lambda payload: (_followup_payload_score(payload), str(payload.get("state_dir") or "")),
        reverse=True,
    )
    return ranked[0] if ranked and _followup_payload_score(ranked[0]) > 0 else None


def _followup_payload_score(payload: Mapping[str, Any]) -> int:
    answer = _raw_case_search_answer(payload.get("steps"))
    if not answer:
        return -1000
    lowered = answer.casefold()
    score = 10
    if "pattern summary:" in lowered:
        score += 80
    if "facebook search-history extraction" in lowered:
        score += 70
    elif "search-history" in lowered:
        score += 35
    if "[l0-" in lowered:
        score += 25
    if "depression demons" in lowered or "maura west suicide" in lowered:
        score += 25
    if "resident synthesis did not return citation-safe prose" in lowered:
        score -= 120
    if "source-excerpt handoff" in lowered or "relevant retrieved excerpts:" in lowered:
        score -= 80
    return score


def _looks_like_completed_followup(text: str) -> bool:
    clean = re.sub(r"[^a-z0-9'\s]+", " ", str(text or "").casefold())
    normalized = " ".join(clean.split())
    words = normalized.split()
    if not normalized:
        return False
    exact = {
        "summarize",
        "summarize it",
        "summarize this",
        "summary",
        "plain summary",
        "make it a summary",
        "what patterns",
        "what are the patterns",
        "pattern summary",
    }
    if normalized in exact:
        return True
    if "not a summary" in normalized or "not what i asked" in normalized:
        return True
    if "non functional" in normalized or "latest response" in normalized or "bad response" in normalized:
        return True
    if normalized.startswith(("tell me what the pattern", "tell me what patterns", "what is she searching")):
        return True
    if "what is she searching for" in normalized or "what was she searching for" in normalized:
        return True
    if normalized in {"what can we do", "what now", "now what"}:
        return True
    return len(words) <= 8 and "patterns" in words


def _looks_like_harness_summary_failure(text: str) -> bool:
    lowered = str(text or "").casefold()
    if "hermes summary:" not in lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "status: failed",
            "no executable plan",
            "no state_dir was recorded",
            "resume: unavailable",
        )
    )


def _orchestrate_request_from_args(args: list[str]) -> str:
    if not args or args[0] != "orchestrate":
        return ""
    if len(args) > 1 and not str(args[1]).startswith("--"):
        return str(args[1])
    return ""


def _flag_value(args: list[str], flag: str) -> str:
    try:
        index = args.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    return str(args[index + 1])


def _normalize_text(text: str) -> str:
    clean = re.sub(r"[^a-z0-9'\s]+", " ", str(text or "").casefold())
    return " ".join(clean.split())


_PLAIN_STOP = frozenset(
    {
        "stop",
        "cancel",
        "nevermind",
        "never mind",
        "hold",
        "pause",
    }
)


def _completed_followup_summary(_request: str, payload: Mapping[str, Any]) -> str:
    answer = _raw_case_search_answer(payload.get("steps"))
    if not answer:
        return ""
    facebook = _facebook_search_history_followup(answer)
    if facebook:
        return facebook[:3900]
    lines = [
        "Plain-English summary of the completed result:",
        one_line(answer, 3200),
    ]
    return "\n".join(lines)[:3900]


def _facebook_search_history_followup(answer: str) -> str:
    if "Facebook search-history" not in answer and "search-history" not in answer:
        return ""
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    pattern_lines = [
        line
        for line in lines
        if line.startswith(
            (
                "Pattern summary:",
                "Repeated query pattern:",
                "Other matching query text:",
                "No matching entries found for:",
            )
        )
    ]
    entries = _facebook_entries_from_answer(lines)
    if pattern_lines:
        out = ["Pattern summary from the completed search-history extraction:", *pattern_lines]
        if entries:
            out.extend(["", "Source-backed entries:"])
            out.extend(f"- {entry}" for entry in entries[:8])
        return "\n".join(out)
    if not entries:
        return "Pattern summary: the completed search-history extraction did not include matching entries."
    by_query: dict[str, list[str]] = {}
    category_counts: dict[str, int] = {}
    for entry in entries:
        category, query, citation = _split_facebook_entry(entry)
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        if query:
            by_query.setdefault(query, []).append(citation)
    count_text = ", ".join(f"{key}={value}" for key, value in category_counts.items())
    out = [
        "Pattern summary from the completed search-history extraction:",
        f"Found {len(entries)} matching mental-health search-history entries" + (f" ({count_text})." if count_text else "."),
    ]
    repeated = [(query, cites) for query, cites in by_query.items() if len(cites) > 1]
    if repeated:
        out.append(
            "Repeated query pattern: "
            + "; ".join(f"{query!r} appears {len(cites)} times ({', '.join(cites)})" for query, cites in repeated)
            + "."
        )
    singles = [(query, cites[0]) for query, cites in by_query.items() if len(cites) == 1]
    if singles:
        out.append("Other matching query text: " + "; ".join(f"{query!r} {cite}" for query, cite in singles) + ".")
    missing = _facebook_missing_categories(lines)
    if missing:
        out.append("No matching entries found for: " + ", ".join(missing) + ".")
    out.extend(["", "Source-backed entries:"])
    out.extend(f"- {entry}" for entry in entries[:8])
    return "\n".join(out)


def _facebook_entries_from_answer(lines: list[str]) -> list[str]:
    entries: list[str] = []
    category = ""
    for line in lines:
        if line.startswith("Depression:"):
            category = "depression"
            continue
        if line.startswith("Suicide:"):
            category = "suicide"
            continue
        if line.startswith("Treatment-resistant depression:"):
            category = "treatment-resistant depression"
            continue
        if line.startswith("Source files reviewed:"):
            category = ""
            continue
        if category and line.startswith("- "):
            entries.append(f"{category}: {line[2:]}")
    return entries


def _split_facebook_entry(entry: str) -> tuple[str, str, str]:
    category, _, rest = entry.partition(":")
    parts = [part.strip() for part in rest.split("|")]
    query = parts[2] if len(parts) >= 3 else rest.strip()
    citation_match = re.search(r"(\[L0-\d+\])", query)
    citation = citation_match.group(1) if citation_match else ""
    query = re.sub(r"\s*\[L0-\d+\]\s*$", "", query).strip()
    return category.strip(), query, citation


def _facebook_missing_categories(lines: list[str]) -> list[str]:
    missing: list[str] = []
    for line in lines:
        if line.endswith(": no matching entries found."):
            missing.append(line.split(":", 1)[0].casefold())
    return missing


def _state_names_from_text(text: str) -> list[str]:
    return re.findall(r"\bState:\s*([^\s]+)", text)


def _prepend_pythonpath(repo_root: Path, existing: str | None) -> str:
    return str(repo_root) if not existing else f"{repo_root}{os.pathsep}{existing}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "matter"


def _ambiguous_message(reference: str, candidates: tuple[str, ...]) -> str:
    return f"The shorthand {reference!r} matched: {', '.join(candidates[:12])}."


def _request_with_memory_context(request: str, context: str) -> str:
    context = context.strip()
    if not context:
        return request
    return "\n\n".join(
        [
            request,
            "Recent Hermes chat memory. Use this only to resolve pronouns, same-client references, pending questions, and operator intent; do not treat it as client evidence:",
            context,
        ]
    )


def safe_error_line(text: str) -> str:
    clean = one_line(str(text or ""), 600)
    clean = clean.replace("/mnt/hdd/Dropbox/Client Files/", "[CLIENT_FILES]/")
    for pattern in _SECRET_PATTERNS:
        clean = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", clean)
    return clean or "unknown error"


def _gateway_model_line() -> str:
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
        if not isinstance(model_cfg, dict):
            return "configured separately"
        model = str(model_cfg.get("default") or model_cfg.get("model") or "configured model")
        base_url = str(model_cfg.get("base_url") or "").strip()
        return f"{model} at {base_url}" if base_url else model
    except Exception:
        return "configured separately"
