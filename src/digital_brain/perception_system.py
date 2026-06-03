from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from digital_brain.models import utc_now


@dataclass(frozen=True)
class SystemPerception:
    timestamp: str
    operating_system: str
    release: str
    cpu_percent: float | None
    ram_percent: float | None
    gpu: str
    network: str
    processes: list[dict[str, Any]]
    open_windows: list[str]
    current_directory: str


class ComputerPerception:
    def observe_system(self) -> SystemPerception:
        cpu_percent: float | None = None
        ram_percent: float | None = None
        processes: list[dict[str, Any]] = []
        try:
            import psutil

            cpu_percent = float(psutil.cpu_percent(interval=0.0))
            ram_percent = float(psutil.virtual_memory().percent)
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                info = proc.info
                processes.append(
                    {
                        "pid": info["pid"],
                        "name": info["name"],
                        "cpu_percent": info["cpu_percent"],
                        "memory_percent": round(float(info["memory_percent"] or 0.0), 4),
                    }
                )
                if len(processes) >= 20:
                    break
        except ImportError:
            processes.append({"pid": os.getpid(), "name": "digital-brain", "cpu_percent": None})

        return SystemPerception(
            timestamp=utc_now(),
            operating_system=platform.system(),
            release=platform.release(),
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            gpu=self._gpu_summary(),
            network=self._network_summary(),
            processes=processes,
            open_windows=self._open_windows(),
            current_directory=str(Path.cwd()),
        )

    def capture_screenshot(self, output_path: str | Path = ".brain/screenshot.png") -> dict[str, Any]:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import mss
            from PIL import Image

            with mss.mss() as screen:
                monitor = screen.monitors[0]
                shot = screen.grab(monitor)
                image = Image.frombytes("RGB", shot.size, shot.rgb)
                image.save(path)
            return {"ok": True, "path": str(path), "timestamp": utc_now()}
        except ImportError as error:
            return {
                "ok": False,
                "path": None,
                "timestamp": utc_now(),
                "error": f"Screenshot capture requires optional dependency: {error.name}",
            }

    def ocr_image(self, image_path: str | Path) -> dict[str, Any]:
        try:
            import pytesseract
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(image_path), lang="tha+eng")
            return {"ok": True, "text": text.strip(), "timestamp": utc_now()}
        except ImportError as error:
            return {
                "ok": False,
                "text": "",
                "timestamp": utc_now(),
                "error": f"OCR requires optional dependency: {error.name}",
            }

    def list_files(self, directory: str | Path) -> list[dict[str, Any]]:
        path = Path(directory).expanduser().resolve()
        items: list[dict[str, Any]] = []
        for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            stat = child.stat()
            items.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "kind": "directory" if child.is_dir() else "file",
                    "size": stat.st_size,
                    "modified_at": stat.st_mtime,
                }
            )
            if len(items) >= 200:
                break
        return items

    def _gpu_summary(self) -> str:
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        return "not detected"

    def _network_summary(self) -> str:
        try:
            import psutil

            counters = psutil.net_io_counters()
            return f"bytes_sent={counters.bytes_sent}, bytes_recv={counters.bytes_recv}"
        except ImportError:
            return "psutil unavailable"

    def _open_windows(self) -> list[str]:
        system = platform.system()
        if system == "Windows":
            command = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object -ExpandProperty MainWindowTitle",
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
            return [line.strip() for line in result.stdout.splitlines() if line.strip()][:50]
        if shutil.which("wmctrl"):
            result = subprocess.run(["wmctrl", "-l"], check=False, capture_output=True, text=True, timeout=5)
            return [line.strip() for line in result.stdout.splitlines() if line.strip()][:50]
        return []
