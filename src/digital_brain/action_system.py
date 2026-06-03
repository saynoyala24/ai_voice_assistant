from __future__ import annotations

import subprocess
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from digital_brain.memory import PersistentMemory
from digital_brain.models import utc_now


@dataclass(frozen=True)
class ActionRequest:
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    approval_token: str | None = None


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    action_type: str
    message: str
    requires_approval: bool = False
    data: dict[str, Any] = field(default_factory=dict)


class SafetyPolicy:
    risky_actions = {
        "write_file",
        "delete_file",
        "run_command",
        "install_program",
        "close_program",
        "mouse_click",
        "keyboard_type",
        "hotkey",
    }
    forbidden_command_terms = {"format", "cipher", "del /f", "rm -rf", "shutdown", "reboot"}

    def __init__(self) -> None:
        self._approvals: set[str] = set()

    def issue_approval(self, action: ActionRequest) -> str:
        token = f"approve-{action.action_type}-{utc_now()}"
        self._approvals.add(token)
        return token

    def check(self, action: ActionRequest) -> ActionResult | None:
        if action.action_type in self.risky_actions and action.approval_token not in self._approvals:
            return ActionResult(
                ok=False,
                action_type=action.action_type,
                message="Action requires approval before execution.",
                requires_approval=True,
                data={"approval_token": self.issue_approval(action)},
            )
        if action.action_type == "run_command":
            command = str(action.parameters.get("command", "")).lower()
            if any(term in command for term in self.forbidden_command_terms):
                return ActionResult(
                    ok=False,
                    action_type=action.action_type,
                    message="Command blocked by safety policy.",
                )
        return None


class LocalActionController:
    def __init__(self, memory: PersistentMemory, safe_root: str | Path = ".") -> None:
        self.memory = memory
        self.safe_root = Path(safe_root).resolve()
        self.safety = SafetyPolicy()

    def execute(self, action: ActionRequest) -> ActionResult:
        safety_result = self.safety.check(action)
        if safety_result:
            self._log(action, safety_result)
            return safety_result

        handlers = {
            "open_url": self._open_url,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "run_command": self._run_command,
            "mouse_click": self._mouse_click,
            "keyboard_type": self._keyboard_type,
            "hotkey": self._hotkey,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            result = ActionResult(False, action.action_type, "Unsupported action type.")
        else:
            result = handler(action.parameters)
        self._log(action, result)
        return result

    def _open_url(self, parameters: dict[str, Any]) -> ActionResult:
        url = str(parameters.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            return ActionResult(False, "open_url", "URL must start with http:// or https://.")
        opened = webbrowser.open(url)
        return ActionResult(opened, "open_url", "Browser open request sent.", data={"url": url})

    def _read_file(self, parameters: dict[str, Any]) -> ActionResult:
        path = self._safe_path(parameters.get("path", ""))
        if path is None or not path.is_file():
            return ActionResult(False, "read_file", "File is outside safe root or does not exist.")
        text = path.read_text(encoding="utf-8", errors="replace")
        return ActionResult(True, "read_file", "File read.", data={"path": str(path), "text": text[:8000]})

    def _write_file(self, parameters: dict[str, Any]) -> ActionResult:
        path = self._safe_path(parameters.get("path", ""))
        if path is None:
            return ActionResult(False, "write_file", "File path is outside safe root.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(parameters.get("content", "")), encoding="utf-8")
        return ActionResult(True, "write_file", "File written.", data={"path": str(path)})

    def _run_command(self, parameters: dict[str, Any]) -> ActionResult:
        command = str(parameters.get("command", "")).strip()
        if not command:
            return ActionResult(False, "run_command", "Command is empty.")
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.safe_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(parameters.get("timeout", 30)),
        )
        return ActionResult(
            result.returncode == 0,
            "run_command",
            "Command completed.",
            data={
                "returncode": result.returncode,
                "stdout": result.stdout[-8000:],
                "stderr": result.stderr[-8000:],
            },
        )

    def _mouse_click(self, parameters: dict[str, Any]) -> ActionResult:
        try:
            import pyautogui

            pyautogui.click(int(parameters.get("x", 0)), int(parameters.get("y", 0)))
            return ActionResult(True, "mouse_click", "Mouse clicked.")
        except ImportError:
            return ActionResult(False, "mouse_click", "pyautogui optional dependency is not installed.")

    def _keyboard_type(self, parameters: dict[str, Any]) -> ActionResult:
        try:
            import pyautogui

            pyautogui.write(str(parameters.get("text", "")), interval=0.01)
            return ActionResult(True, "keyboard_type", "Text typed.")
        except ImportError:
            return ActionResult(False, "keyboard_type", "pyautogui optional dependency is not installed.")

    def _hotkey(self, parameters: dict[str, Any]) -> ActionResult:
        try:
            import pyautogui

            keys = [str(key) for key in parameters.get("keys", [])]
            if not keys:
                return ActionResult(False, "hotkey", "No keys supplied.")
            pyautogui.hotkey(*keys)
            return ActionResult(True, "hotkey", "Hotkey sent.", data={"keys": keys})
        except ImportError:
            return ActionResult(False, "hotkey", "pyautogui optional dependency is not installed.")

    def _safe_path(self, value: Any) -> Path | None:
        path = (self.safe_root / str(value)).resolve()
        if path == self.safe_root or self.safe_root in path.parents:
            return path
        return None

    def _log(self, action: ActionRequest, result: ActionResult) -> None:
        self.memory.log_event(
            "action",
            result.message,
            {
                "action_type": action.action_type,
                "parameters": action.parameters,
                "ok": result.ok,
                "requires_approval": result.requires_approval,
                "data": result.data,
            },
        )
