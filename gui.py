import asyncio
import gradio as gr
import logfire
import sys
import time
from dotenv import load_dotenv
from missions import execute_multi_sheet_mission, execute_analysis_mission, execute_sql_mission
from lib.ui_bridge import UIBridge, PhaseUpdate

# Load environment and configure
load_dotenv(override=True)
logfire.configure()

PHASE_PROGRESS = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0}
# Throttle UI updates so the log doesn't flicker or overwrite (max ~6/sec)
LOG_UPDATE_INTERVAL = 0.18


async def run_analysis(file_obj, sql_conn, sheet_name, progress=gr.Progress()):
    queue = asyncio.Queue()
    bridge = UIBridge(queue)

    if not (file_obj or sql_conn):
        yield "⚠️ Please provide a file or a SQL connection string.", "", None
        return

    source = sql_conn if sql_conn else file_obj.name

    # Run mission as a background task (CPU-bound work runs in threads inside missions)
    if sql_conn:
        mission_task = asyncio.create_task(execute_sql_mission(source, bridge=bridge))
    elif sheet_name:
        mission_task = asyncio.create_task(execute_analysis_mission(source, target_sheet=sheet_name, bridge=bridge))
    else:
        mission_task = asyncio.create_task(execute_multi_sheet_mission(source, bridge=bridge))

    status_lines = ["🚀 Mission Started..."]
    last_progress_value = 0.0
    last_progress_desc = ""
    last_yield_time = 0.0
    pending_update = False

    print(f"\n[TERMINAL LOG] Mission started for {source}")
    sys.stdout.flush()

    while not mission_task.done():
        try:
            update = await asyncio.wait_for(queue.get(), timeout=0.12)
            if isinstance(update, PhaseUpdate):
                icon = "⏳" if update.status == "running" else "✅"
                new_line = f"{icon} [{update.label}] {update.detail}"
                status_lines.append(new_line)
                print(f"[UI UPDATE] {new_line}")
                sys.stdout.flush()

                pval = PHASE_PROGRESS.get(update.phase, 0.5)
                pdesc = f"Phase {update.phase}: {update.label}"
                if pval != last_progress_value or pdesc != last_progress_desc:
                    progress(pval, desc=pdesc)
                    last_progress_value, last_progress_desc = pval, pdesc
                pending_update = True
        except asyncio.TimeoutError:
            pass

        now = time.monotonic()
        if pending_update and (now - last_yield_time) >= LOG_UPDATE_INTERVAL:
            status_log = "\n".join(status_lines)
            yield status_log, "", None
            last_yield_time = now
            pending_update = False

    # Drain any remaining updates
    while True:
        try:
            update = queue.get_nowait()
            if isinstance(update, PhaseUpdate):
                icon = "⏳" if update.status == "running" else "✅"
                status_lines.append(f"{icon} [{update.label}] {update.detail}")
        except asyncio.QueueEmpty:
            break

    try:
        result = await mission_task
        status_log = "\n".join(status_lines)
        if result:
            narrative = result.get("narrative", "")
            pptx_path = result.get("pptx_path")
            status_log += "\n\n🏁 Mission accomplished successfully."
            print("[TERMINAL LOG] Mission accomplished successfully.")
            sys.stdout.flush()
            yield status_log, narrative, pptx_path
        else:
            status_log += "\n\n❌ Mission failed."
            yield status_log, "No results.", None
    except Exception as e:
        status_log = "\n".join(status_lines) + f"\n\n❌ Critical Error: {str(e)}"
        print(f"[ERROR] {str(e)}")
        sys.stdout.flush()
        yield status_log, "Error.", None

# --- UI Layout ---
with gr.Blocks(title="Agentic Data Analyst", theme="soft") as demo:
    gr.Markdown("# 🧠 Agentic Data Analyst")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Dataset")
            sql_input = gr.Textbox(label="OR SQL Connection")
            sheet_input = gr.Textbox(label="Target Sheet (Optional)")
            analyze_btn = gr.Button("🚀 Start Mission", variant="primary")
            
        with gr.Column(scale=2):
            # Using Textbox with autoscroll prevents text overlap and mixing
            status_output = gr.Textbox(
                label="Mission Progress Log", 
                lines=15, 
                max_lines=30, 
                interactive=False, 
                autoscroll=True
            )
            
    with gr.Row():
        with gr.Column():
            report_output = gr.Markdown()
        with gr.Column():
            file_output = gr.File(label="Download PPTX")

    analyze_btn.click(
        fn=run_analysis,
        inputs=[file_input, sql_input, sheet_input],
        outputs=[status_output, report_output, file_output],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch()