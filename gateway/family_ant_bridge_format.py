"""Reply formatting helpers for the Family Ant gateway bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def summarize_payload(payload: Mapping[str, Any]) -> str:
    plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
    status = str(payload.get("status") or "unknown")
    state_name = Path(str(payload.get("state_dir") or "")).name or "none"
    questions = payload_questions(payload)
    workflows = workflow_labels(plan.get("steps"))
    manual = _manual_boundary_labels(workflows)
    action = _plain_action_summary(plan, workflows)
    lines = ["Analysis:"]
    if questions:
        lines.append(
            "I can handle this request, but I am missing a concrete detail that is material to routing it safely."
        )
    elif status == "dry_run" and manual:
        lines.append(
            f"I have a saved executable plan to {action}, but I am holding auto-run because it crosses a "
            "service, delivery, court-facing finalization/export, or external account-sync boundary."
        )
    elif status == "dry_run":
        lines.append(f"I have a saved executable plan to {action}.")
    elif status in {"completed", "needs_review"}:
        lines.append(f"Hermes returned a reviewable result for the plan to {action}.")
    elif status == "failed":
        lines.append(
            "Hermes failed before it returned a reliable legal-workflow answer. "
            "I preserved the diagnostic state and am treating this as an orchestration failure, not attorney work product."
        )
    else:
        lines.append(
            f"Hermes returned `{status}` for the plan to {action}. I am treating that as workflow posture, "
            "not as a legal conclusion."
        )
    summary = one_line(str(plan.get("summary") or ""), 420)
    if summary:
        lines.extend(["", "Plan:", "- " + summary])
    if workflows:
        lines.append("- Steps: " + ", ".join(workflows))
    if state_name != "none":
        lines.append("- Saved state: " + state_name)
    diagnostics = diagnostic_lines(payload)
    if diagnostics:
        lines.extend(["", "Diagnostics:"])
        lines.extend(diagnostics)
    if questions:
        lines.extend(["", "Needed detail:"])
        lines.extend(f"- {name}: {question}" for name, question in questions[:8])
    artifacts = artifact_names(payload.get("steps"))
    if artifacts:
        lines.extend(["", "Source support:", "- Artifacts reported: " + ", ".join(artifacts) + "."])
    answer = case_search_answer(payload.get("steps"))
    if answer:
        if "Source support:" not in lines:
            lines.extend(["", "Source support:"])
        lines.append(answer)
    lines.extend(["", "Limits:"])
    if questions:
        lines.append("- I have not run the legal workflow yet because the missing detail is needed for a reliable route.")
    elif status == "dry_run" and manual:
        lines.append(
            "- This is a plan posture only. Telegram does not auto-run service, delivery, court-facing finalization/export, or external account-sync workflows."
        )
    elif status == "dry_run":
        lines.append("- This is a plan posture only; no artifact or legal conclusion has been produced by this reply.")
    elif status == "failed":
        lines.append("- No artifact or legal conclusion was produced by this failed orchestration summary.")
    else:
        lines.append("- This Telegram summary does not substitute for opening the artifact or source record.")
    lines.extend(["", "Next:"])
    if questions:
        lines.append("- Reply in normal language with the missing detail; Hermes will resume from the saved context when it can.")
    elif status == "dry_run" and manual:
        lines.append(
            "- Review the planned workflow, documents, recipients, and external-action boundary before running it manually from the saved state."
        )
    elif status == "dry_run":
        lines.append("- I will run safe registered workflows automatically from normal Telegram requests; this saved plan remains available for manual execution.")
    elif status == "failed":
        lines.append("- Fix the diagnostic blocker or retry the same request; the saved state is available for debugging.")
    else:
        lines.append("- Review the source-backed result or artifact before relying on it outside the firm.")
    return "\n".join(lines)[:3900]


def payload_questions(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    validation = payload.get("validation") if isinstance(payload.get("validation"), Mapping) else {}
    raw = validation.get("questions") if isinstance(validation.get("questions"), list) else []
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, Mapping):
            name = str(item.get("input_name") or "input")
            question = one_line(str(item.get("question") or ""), 320)
            out.append((name, question))
    return out


def workflow_labels(steps: Any) -> list[str]:
    labels: list[str] = []
    if isinstance(steps, list):
        for item in steps:
            if isinstance(item, Mapping):
                if item.get("kind") == "case_search":
                    label = "case_search"
                else:
                    label = str(item.get("workflow") or item.get("tool") or item.get("kind") or "").strip()
                if label:
                    labels.append(label)
    return labels


def _plain_action_summary(plan: Mapping[str, Any], workflows: list[str]) -> str:
    if workflows == ["case_search"]:
        return "search the local case file and return a source-supported analysis"
    if workflows:
        return "run " + ", ".join(workflows)
    summary = one_line(str(plan.get("summary") or ""), 220)
    if "failed before it could build" in summary.casefold():
        return "diagnose the failed Hermes orchestration"
    return summary or "run the requested local Hermes task"


def _manual_boundary_labels(workflows: list[str]) -> list[str]:
    manual: list[str] = []
    for label in workflows:
        normalized = str(label or "").strip().casefold().replace("-", "_")
        if normalized in {
            "finalization_export_review_gate",
            "gmail_ingest",
            "service_email",
            "trial_package_delivery",
        }:
            manual.append(label)
    return manual


def case_search_answer(steps: Any) -> str:
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if step.get("kind") != "case_search" and step.get("tool") != "case_search":
            continue
        payload = step.get("payload") if isinstance(step.get("payload"), Mapping) else {}
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            continue
        raw_sources = payload.get("sources")
        source_count = len(raw_sources) if isinstance(raw_sources, list) else 0
        review = str(payload.get("review_status") or step.get("terminal_status") or "NEEDS_REVIEW")
        prefix = f"Answer ({review}, {source_count} sources): "
        return prefix + one_line(answer, 2400)
    return ""


def artifact_names(steps: Any) -> list[str]:
    names: list[str] = []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            for raw in step.get("artifact_paths", []):
                name = Path(str(raw)).name
                if name:
                    names.append(name)
    return list(dict.fromkeys(names))


def diagnostic_lines(payload: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    advice = payload.get("failure_advice")
    if isinstance(advice, Mapping):
        text = one_line(str(advice.get("text") or advice.get("error") or ""), 900)
        if text:
            status = one_line(str(advice.get("status") or ""), 80)
            prefix = f"Codex diagnostic ({status}): " if status else "Codex diagnostic: "
            lines.append("- " + prefix + text)
    error = one_line(str(payload.get("error") or ""), 500)
    if error:
        lines.append("- Harness error: " + error)
    validation = payload.get("validation") if isinstance(payload.get("validation"), Mapping) else {}
    raw_errors = validation.get("errors") if isinstance(validation.get("errors"), list) else []
    if not raw_errors and isinstance(payload.get("validation_errors"), list):
        raw_errors = payload.get("validation_errors")
    seen: set[str] = set()
    for item in raw_errors[:4]:
        line = one_line(str(item or ""), 500)
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append("- Validation: " + line)
    return lines


def one_line(text: str, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: max(0, limit - 3)].rstrip() + "..."
