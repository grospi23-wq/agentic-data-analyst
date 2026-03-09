from dataclasses import dataclass
from typing import Literal
import asyncio

@dataclass
class PhaseUpdate:
    """
    Represents a status update from the analysis pipeline.
    Used to bridge the gap between async execution and Gradio's UI.
    """
    phase: int
    label: str
    status: Literal["running", "done", "error"]
    detail: str = ""

class UIBridge:
    """
    A messaging bridge that allows the mission logic to 'push'
    updates into an asyncio.Queue for the UI to consume.
    The UI consumer should flush stdout after handling updates so that
    terminal and Logfire output stay in sync with mission progress.
    """
    def __init__(self, queue: asyncio.Queue | None = None):
        self.queue = queue

    async def emit(self, phase: int, label: str, status: Literal["running", "done", "error"], detail: str = ""):
        """
        Sends an update to the UI queue if it exists.
        """
        if self.queue:
            update = PhaseUpdate(phase=phase, label=label, status=status, detail=detail)
            await self.queue.put(update)
