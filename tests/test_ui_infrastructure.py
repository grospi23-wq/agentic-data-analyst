import asyncio
import pytest
import threading
from lib.ui_bridge import UIBridge, PhaseUpdate

@pytest.mark.asyncio
async def test_ui_bridge_emission():
    """
    Verifies that messages sent via the Bridge correctly populate the Queue.
    Ensures the data structure remains consistent for Gradio consumption.
    """
    queue = asyncio.Queue()
    bridge = UIBridge(queue)

    # Emit a sample phase update
    await bridge.emit(2, "Strategy", "running", "Defining mission strategy")

    # Retrieve from queue
    update = await queue.get()

    assert isinstance(update, PhaseUpdate)
    assert update.phase == 2
    assert update.label == "Strategy"
    assert update.status == "running"
    assert "Defining mission strategy" in update.detail

@pytest.mark.asyncio
async def test_ui_bridge_thread_safety():
    """
    Crucial test: Verifies the Bridge can receive updates from a background thread.
    This simulates heavy Pandas operations or file I/O offloaded from the main loop.
    """
    queue = asyncio.Queue()
    bridge = UIBridge(queue)
    loop = asyncio.get_running_loop()

    def sync_task_in_background_thread():
        # Simulate a synchronous process (like a Join) reporting back to the UI
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(
                bridge.emit(3, "Joins", "done", "Threaded operation finished successfully")
            )
        )

    # Launch the mock background worker
    worker_thread = threading.Thread(target=sync_task_in_background_thread)
    worker_thread.start()
    worker_thread.join()

    # Wait for the Event Loop to process the thread-safe call
    update = await asyncio.wait_for(queue.get(), timeout=2.0)

    assert update.phase == 3
    assert "Threaded operation finished" in update.detail
    assert update.status == "done"
