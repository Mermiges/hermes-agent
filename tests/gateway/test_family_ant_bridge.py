from __future__ import annotations

from datetime import datetime
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.family_ant_bridge import HarnessGatewayBridge, MatterTarget
from gateway.family_ant_bridge_format import summarize_payload
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        user_name="User",
        chat_id="c1",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="m1")


def _runner(fake_bridge):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="***",
                extra={},
            )
        }
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._family_ant_harness_bridge_impl = fake_bridge
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:c1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


class FakeBridge:
    def __init__(self, *, saved_plan: bool = False, pending_questions: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.saved_plan = saved_plan
        self.pending_questions = pending_questions

    async def plan(self, session_key: str, request: str) -> str:
        self.calls.append(("plan", session_key, request))
        self.saved_plan = True
        return "Status: dry_run\nPlan steps: analyze\nNext: send /go to execute this plan."

    async def go(self, session_key: str) -> str:
        self.calls.append(("go", session_key, ""))
        self.pending_questions = False
        return "Status: completed\nArtifacts: report.md"

    async def answer(self, session_key: str, answer_text: str) -> str:
        self.calls.append(("answer", session_key, answer_text))
        self.pending_questions = False
        self.saved_plan = True
        return "Status: dry_run\nNext: send /go to execute this plan."

    def brain(self, raw_name: str) -> str:
        self.calls.append(("brain", "", raw_name))
        return "Current Family Ant router brain: codex_cli"

    async def status(self) -> str:
        self.calls.append(("status", "", ""))
        return "Fleet health: 3 up, 0 down"

    def has_saved_plan(self, session_key: str) -> bool:
        return self.saved_plan

    def has_pending_questions(self, session_key: str) -> bool:
        return self.pending_questions


class FailingBridge(FakeBridge):
    async def plan(self, session_key: str, request: str) -> str:
        self.calls.append(("plan", session_key, request))
        raise RuntimeError(
            "Hermes summary: - status: failed - planned: no executable plan "
            "- ran: nothing - produced: no artifact paths reported"
        )


class StubHarnessBridge(HarnessGatewayBridge):
    def __init__(self) -> None:
        super().__init__(hermes_home=Path(tempfile.mkdtemp(prefix="hermes-test-home-")))
        self.commands: list[list[str]] = []

    def _resolve_matter(self, request: str, *, session_key: str) -> MatterTarget:
        return MatterTarget("chipman-chris", Path("/tmp/chipman"), request)

    async def _run_orchestrate(self, args: list[str], *, timeout: int) -> dict:
        self.commands.append(args)
        return {
            "status": "dry_run",
            "state_dir": "/repo/runs/hermes-orchnl/state-1",
            "plan": {
                "summary": "Plan summary only.",
                "steps": [{"workflow": "analyze"}],
            },
            "validation": {"questions": []},
            "steps": [],
        }

    def _associate_response(self, session_key: str, *, request: str, run_payload: dict, plan_payload: dict | None = None) -> str:
        return summarize_payload(run_payload)


class ManualBoundaryBridge(StubHarnessBridge):
    async def _run_orchestrate(self, args: list[str], *, timeout: int) -> dict:
        self.commands.append(args)
        return {
            "status": "dry_run",
            "state_dir": "/repo/runs/hermes-orchnl/state-service",
            "plan": {
                "summary": "Prepare an AIS email-service handoff for attorney review.",
                "steps": [{"workflow": "service_email"}],
            },
            "validation": {"questions": []},
            "steps": [],
        }


class CompletedAutoRunBridge(StubHarnessBridge):
    async def _run_orchestrate(self, args: list[str], *, timeout: int) -> dict:
        self.commands.append(args)
        if len(self.commands) == 1:
            return {
                "status": "dry_run",
                "state_dir": "/repo/runs/hermes-orchnl/state-case-search",
                "plan": {
                    "summary": "Search the local case file.",
                    "steps": [{"kind": "case_search", "tool": "case_search"}],
                },
                "validation": {"questions": []},
                "steps": [],
            }
        return {
            "status": "completed",
            "state_dir": "/repo/runs/hermes-orchnl/state-case-search",
            "plan": {
                "summary": "Search the local case file.",
                "steps": [{"kind": "case_search", "tool": "case_search"}],
            },
            "validation": {"questions": []},
            "steps": [
                {
                    "kind": "case_search",
                    "tool": "case_search",
                    "status": "completed",
                    "terminal_status": "NEEDS_REVIEW",
                    "payload": {
                        "answer": "Found source-backed depression search-history references [S1].",
                        "review_status": "NEEDS_REVIEW",
                        "sources": [{"source_id": "S1"}],
                    },
                }
            ],
        }


class IntakeQuestionBridge(StubHarnessBridge):
    def _resolve_matter(self, request: str, *, session_key: str) -> MatterTarget:
        if ":" not in request:
            from gateway.family_ant_bridge import BridgeUserQuestion

            raise BridgeUserQuestion("Which client folder should Hermes use?")
        return MatterTarget("chipman-chris", Path("/tmp/chipman"), request)


class FailedJsonSubprocessBridge(HarnessGatewayBridge):
    def __init__(self) -> None:
        super().__init__(hermes_home=Path(tempfile.mkdtemp(prefix="hermes-test-home-")))
        self.ok_returncodes: tuple[int, ...] | None = None

    def _resolve_matter(self, request: str, *, session_key: str) -> MatterTarget:
        return MatterTarget("chipman-chris", Path("/tmp/chipman"), request)

    async def _run(
        self,
        command: list[str],
        *,
        timeout: int,
        ok_returncodes: tuple[int, ...] = (0, 2),
    ) -> tuple[str, str]:
        self.ok_returncodes = ok_returncodes
        return json.dumps(_failed_harness_payload()), ""


class NonJsonSummaryBridge(HarnessGatewayBridge):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(
            repo_root=repo_root,
            hermes_home=Path(tempfile.mkdtemp(prefix="hermes-test-home-")),
        )
        self.ok_returncodes: tuple[int, ...] | None = None

    def _resolve_matter(self, request: str, *, session_key: str) -> MatterTarget:
        return MatterTarget("chipman-chris", Path("/tmp/chipman"), request)

    async def _run(
        self,
        command: list[str],
        *,
        timeout: int,
        ok_returncodes: tuple[int, ...] = (0, 2),
    ) -> tuple[str, str]:
        self.ok_returncodes = ok_returncodes
        return "", (
            "Hermes summary:\n"
            "- status: failed\n"
            "- planned: no executable plan\n"
            "- ran: nothing\n"
            "- produced: no artifact paths reported\n"
            "- codex_help: not requested\n"
            "- resume: unavailable; no state_dir was recorded"
        )

    def _codex_failure_advice(self, payload: dict, *, state_dir: Path) -> dict:
        return {
            "status": "completed",
            "text": "root cause: gateway received non-JSON summary; retry through CLI JSON boundary.",
        }


def test_bridge_plan_builds_dry_run_command_and_remembers_state():
    bridge = StubHarnessBridge()
    rendered = asyncio.run(bridge.plan("session-1", "Chipman: summarize status"))
    assert rendered.startswith("Analysis:")
    assert "Next: send go" not in rendered
    assert bridge.sessions["session-1"].state_dir.endswith("state-1")
    assert bridge.sessions["session-1"].pending_request == "Chipman: summarize status"
    assert bridge.commands[0][0] == "orchestrate"
    assert bridge.commands[0][1].splitlines()[0] == "Chipman: summarize status"
    assert "Recent Hermes chat memory" in bridge.commands[0][1]
    assert bridge.commands[0][2:6] == [
        "--dry-run",
        "--matter-id",
        "chipman-chris",
        "--matter-path",
    ]
    assert bridge.commands[1] == ["orchestrate", "--resume", "/repo/runs/hermes-orchnl/state-1"]


def test_bridge_does_not_auto_run_manual_service_boundary():
    bridge = ManualBoundaryBridge()
    rendered = asyncio.run(bridge.plan("session-1", "Chipman: serve final packet by AIS email"))

    assert len(bridge.commands) == 1
    assert bridge.commands[0][0] == "orchestrate"
    assert "--dry-run" in bridge.commands[0]
    assert rendered.startswith("Analysis:")
    assert "- Steps: service_email" in rendered
    assert "Status:" not in rendered
    assert "State:" not in rendered
    assert "will not auto-run service, delivery" in rendered
    assert "Review the documents, recipients" in rendered
    assert "I can run this now unless you say stop" not in rendered


def test_bridge_completed_auto_run_says_what_it_will_do_first():
    bridge = CompletedAutoRunBridge()
    rendered = asyncio.run(bridge.plan("session-1", "Chipman: search history for depression"))

    assert rendered.startswith("I will search the local case file and return a source-supported analysis.")
    assert "\nAnalysis:" in rendered
    assert "Found source-backed depression search-history references [S1]" in rendered
    assert "Next: send go" not in rendered


def test_plain_answer_resumes_pending_intake_question():
    bridge = IntakeQuestionBridge()
    first = asyncio.run(bridge.plan("session-1", "summarize search history"))
    second = asyncio.run(bridge.answer("session-1", "Chipman Chris"))

    assert "reply normally" in first
    assert second.startswith("Analysis:")
    assert "Next: send go" not in second
    assert bridge.commands[0][1].splitlines()[0] == "Chipman Chris: summarize search history"
    assert "Recent Hermes chat memory" in bridge.commands[0][1]


def test_bridge_uses_persistent_chat_memory_for_same_client_request(tmp_path):
    bridge = HarnessGatewayBridge(hermes_home=tmp_path)
    bridge._chat_memory().update_session(
        "session-1",
        matter_id="chipman-chris",
        matter_path=str(tmp_path),
        has_questions=False,
    )

    target = bridge._resolve_matter("same client summarize search history", session_key="session-1")

    assert target.matter_id == "chipman-chris"
    assert target.matter_path == tmp_path


def test_bridge_ignores_closed_persistent_chat_memory(tmp_path):
    from gateway.family_ant_bridge import BridgeUserQuestion

    closed_matter = tmp_path / "!Closed" / "Inglesby Joel"
    closed_matter.mkdir(parents=True)
    bridge = HarnessGatewayBridge(hermes_home=tmp_path)
    bridge._chat_memory().update_session(
        "session-1",
        matter_id="inglesby-joel",
        matter_path=str(closed_matter),
        has_questions=False,
    )

    with pytest.raises(BridgeUserQuestion):
        bridge._resolve_matter("same client summarize search history", session_key="session-1")


def test_bridge_sets_bounded_failure_advisor_timeout(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_CODEX_FAILURE_ADVISOR_TIMEOUT", raising=False)
    bridge = HarnessGatewayBridge(hermes_home=tmp_path)

    env = bridge._subprocess_env()

    assert env["HERMES_CODEX_FAILURE_ADVISOR"] == "1"
    assert env["HERMES_CODEX_FAILURE_ADVISOR_TIMEOUT"] == "45"


def test_request_with_memory_context_preserves_first_line_for_intake():
    from gateway.family_ant_bridge import _request_with_memory_context

    rendered = _request_with_memory_context("Chipman summarize records", "last_matter: chipman")

    assert rendered.splitlines()[0] == "Chipman summarize records"
    assert "Recent Hermes chat memory" in rendered


def test_formatter_includes_case_search_answer():
    rendered = summarize_payload(
        {
            "status": "completed",
            "state_dir": "/repo/runs/hermes-orchnl/state-1",
            "plan": {
                "summary": "Search the matter.",
                "steps": [{"kind": "case_search", "workflow": "analyze", "tool": "case_search"}],
            },
            "validation": {"questions": []},
            "steps": [
                {
                    "kind": "case_search",
                    "tool": "case_search",
                    "status": "completed",
                    "terminal_status": "NEEDS_REVIEW",
                    "payload": {
                        "answer": "Found depression references [S1].",
                        "review_status": "NEEDS_REVIEW",
                        "sources": [{"source_id": "S1"}],
                    },
                }
            ],
        }
    )

    assert "- Steps: case_search" in rendered
    assert "Answer (NEEDS_REVIEW, 1 sources): Found depression references [S1]." in rendered
    assert "Plan steps:" not in rendered
    assert "Status:" not in rendered


def _failed_harness_payload() -> dict:
    return {
        "status": "failed",
        "state_dir": "/repo/runs/hermes-orchnl/state-failed",
        "plan": {
            "summary": "Hermes failed before it could build an executable plan.",
            "steps": [],
        },
        "validation": {
            "questions": [],
            "errors": ["cli_orchestrate_failed:RuntimeError: router setup exploded"],
        },
        "steps": [],
        "failure_advice": {
            "status": "completed",
            "text": "root cause: router setup exploded; immediate retry/fix: inspect config.",
        },
    }


def test_formatter_includes_failure_advice():
    rendered = summarize_payload(_failed_harness_payload())

    assert "Hermes failed before it returned a reliable legal-workflow answer" in rendered
    assert "Diagnostics:" in rendered
    assert "Codex diagnostic (completed): root cause: router setup exploded" in rendered
    assert "Validation: cli_orchestrate_failed:RuntimeError: router setup exploded" in rendered
    assert "No artifact or legal conclusion was produced" in rendered


def test_bridge_accepts_failed_json_payload_from_harness_returncode():
    bridge = FailedJsonSubprocessBridge()
    rendered = asyncio.run(bridge.plan("session-1", "Chipman: update profile and wiki"))

    assert bridge.ok_returncodes == (0, 1, 2)
    assert "Hermes failed before it returned a reliable legal-workflow answer" in rendered
    assert "Codex diagnostic (completed): root cause: router setup exploded" in rendered
    assert "I will not run this automatically" in rendered
    assert "I can run this now unless" not in rendered
    assert bridge.sessions["session-1"].pending_request == "Chipman: update profile and wiki"


def test_bridge_synthesizes_state_for_non_json_no_plan_summary(tmp_path: Path):
    bridge = NonJsonSummaryBridge(tmp_path)
    rendered = asyncio.run(bridge.plan("session-1", "Chipman: update profile and wiki"))

    assert bridge.ok_returncodes == (0, 1, 2)
    assert "Hermes failed before it returned a reliable legal-workflow answer" in rendered
    assert "Saved state: gateway-failure-" in rendered
    assert "Codex diagnostic (completed): root cause: gateway received non-JSON summary" in rendered
    assert "gateway_orchestrate_failed" in rendered
    assert "I will not run this automatically" in rendered
    assert "Harness command failed" not in rendered
    assert "no state_dir was recorded" not in rendered
    session = bridge.sessions["session-1"]
    assert session.state_dir
    state_path = Path(session.state_dir) / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["failure_advice"]["status"] == "completed"
    assert state["plan"]["summary"] == "Hermes failed before it could build an executable plan."


def test_bridge_plan_writes_prompt_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "events.jsonl"
    monkeypatch.setenv("HERMES_PROMPT_LEDGER", str(ledger))
    bridge = StubHarnessBridge()

    asyncio.run(bridge.plan("session-1", "Chipman: summarize status"))

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == [
        "telegram_inbound",
        "telegram_plan_result",
        "telegram_auto_run_result",
    ]
    assert rows[0]["text"] == "Chipman: summarize status"
    assert rows[1]["payload_summary"]["status"] == "dry_run"


def test_harness_command_dispatches_to_bridge():
    bridge = FakeBridge()
    result = asyncio.run(
        _runner(bridge)._handle_message(_event("/harness Chipman: summarize status"))
    )
    assert "Status: dry_run" in result
    assert bridge.calls[0] == (
        "plan",
        "agent:main:telegram:dm:c1",
        "Chipman: summarize status",
    )


def test_plain_telegram_message_dispatches_to_harness_bridge():
    bridge = FakeBridge()
    result = asyncio.run(
        _runner(bridge)._handle_message(_event("Chipman: summarize status"))
    )
    assert "Status: dry_run" in result
    assert bridge.calls[0] == (
        "plan",
        "agent:main:telegram:dm:c1",
        "Chipman: summarize status",
    )


def test_plain_harness_failure_is_associate_style_not_raw_runtime_error():
    bridge = FailingBridge()
    result = asyncio.run(
        _runner(bridge)._handle_message(_event("Chipman: update profile and wiki"))
    )

    assert "Analysis:" in result
    assert "router did not produce an executable Family Ant workflow" in result
    assert "No client files were changed" in result
    assert "You do not need to use /answer" in result
    assert "Diagnostic: RuntimeError:" in result
    assert "Harness command failed" not in result
    assert bridge.calls[0] == (
        "plan",
        "agent:main:telegram:dm:c1",
        "Chipman: update profile and wiki",
    )


def test_plain_go_dispatches_saved_harness_plan():
    bridge = FakeBridge(saved_plan=True)
    result = asyncio.run(_runner(bridge)._handle_message(_event("run it")))
    assert "completed" in result
    assert bridge.calls[0] == ("go", "agent:main:telegram:dm:c1", "")


def test_plain_answer_dispatches_pending_harness_question():
    bridge = FakeBridge(saved_plan=True, pending_questions=True)
    result = asyncio.run(_runner(bridge)._handle_message(_event("doc type is motion")))
    assert "dry_run" in result
    assert bridge.calls[0] == (
        "answer",
        "agent:main:telegram:dm:c1",
        "doc type is motion",
    )


def test_loop_commands_dispatch_to_saved_bridge_state():
    bridge = FakeBridge()
    runner = _runner(bridge)
    assert "completed" in asyncio.run(runner._handle_message(_event("/go")))
    assert "dry_run" in asyncio.run(
        runner._handle_message(_event("/answer doc type is motion"))
    )
    assert "codex_cli" in asyncio.run(runner._handle_message(_event("/brain codex")))
    assert "Fleet health" in asyncio.run(runner._handle_message(_event("/hstatus")))
    assert ("go", "agent:main:telegram:dm:c1", "") in bridge.calls
    assert ("answer", "agent:main:telegram:dm:c1", "doc type is motion") in bridge.calls
    assert ("brain", "", "codex") in bridge.calls
    assert ("status", "", "") in bridge.calls


def test_command_registry_knows_family_ant_gateway_commands():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    for name in ("harness", "go", "answer", "brain", "hstatus"):
        command = resolve_command(name)
        assert command is not None
        assert command.gateway_only
        assert name in GATEWAY_KNOWN_COMMANDS


def test_bridge_formatter_reports_filenames_only():
    payload = {
        "status": "completed",
        "state_dir": "/repo/runs/hermes-orchnl/20260705T010101Z",
        "plan": {
            "summary": "Plan summary only.",
            "steps": [{"workflow": "analyze"}],
        },
        "validation": {"questions": []},
        "steps": [
            {
                "artifact_paths": [
                    "/mnt/hdd/Dropbox/Client Files/Client Name/!lf/md-output/report.md",
                ]
            }
        ],
    }
    rendered = summarize_payload(payload)
    assert "report.md" in rendered
    assert "/mnt/hdd/Dropbox" not in rendered
    assert "Client Name" not in rendered
