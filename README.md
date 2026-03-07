# 🧠 Agentic Data Analyst: Autonomous Relational Data Engine

![CI](https://img.shields.io/github/actions/workflow/status/grospi23-wq/agentic-data-analyst/ci.yml?label=CI&style=flat)
![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Pydantic-AI](https://img.shields.io/badge/Framework-Pydantic--AI-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458)
![SQLite](https://img.shields.io/badge/Memory-aiosqlite-003B57)
![Logfire](https://img.shields.io/badge/Observability-Logfire-red)

An enterprise-grade, multi-agent AI system designed to autonomously profile, analyze, and visualize complex relational datasets (Excel/CSV).

## 📺 Live Demo: Northwind Analysis
![Autonomous Data Analysis Demo](./demo.svg)

> **Note:** This recording demonstrates the system resolving a dataset from the Windows Downloads folder, identifying 9 table relationships, and performing complex multi-sheet joins using GPT-4o and Claude 3.5 Sonnet.

## 🎯 The Problem it Solves
Standard LLM-based analysis often fails on real-world relational data because it:
1. **Hallucinates table relationships** and column names.
2. **Exceeds context windows** by attempting to process raw data rows.
3. **Repeats analytical mistakes** due to a lack of memory between runs.
4. **Produces static, uneditable images** for charts that cannot be adjusted by business users.

## 🏗️ Core Architectural Innovations

### 1. Deterministic Discovery & Join Planning
The `data_discovery_lib` serves as a non-LLM preprocessing layer. It calculates value overlap and cardinality via statistical sampling to generate high-confidence join hints (e.g., `one_to_many`, `left join`). The LLM only receives this lightweight `GlobalDiscoveryMap`, ensuring structural integrity.

### 2. Persistent Memory & Schema Fingerprinting
Utilizing `aiosqlite` and `memory_manager.py`, the system generates a stable SHA-256 hash of the dataset's structure. This allows the engine to:
* **Recall validated joins:** Reuse successful configurations from prior runs.
* **Calibrate via Critic warnings:** Inject historical data-quality warnings directly into the prompt to prevent the re-escalation of known patterns.

### 3. "Trust but Verify" Python Execution
The Analyst agent is equipped with a custom `execute_python_analysis` tool. Before claiming an insight, it writes and runs isolated Pandas code against the live data to verify mathematical claims, Pareto concentrations, or correlation signs.

### 4. Dynamic Domain Specialist Routing
The system automatically classifies the dataset into domains such as RETAIL, FINANCE, SPORTS, or HEALTHCARE. This drives a "Specialist Agency" model where the Analyst adopts domain-specific frameworks (e.g., Pareto 80/20, Bimodal Distribution Detection, Paradox Flagging).

### 5. Native PowerPoint Rendering via DTOs
Following strict DTO (Data Transfer Object) patterns, the LLM outputs a `PresentationSpec`. A deterministic rendering engine then converts this into native, editable PowerPoint objects using `python-pptx`.

### 6. Universal Path Resolver (WSL/Windows Bridge)
To eliminate manual path configuration, the engine implements a dynamic resolver that scans both WSL and Windows host environments. It automatically locates datasets in common landing zones like `Downloads`, `Desktop`, and `Documents`, ensuring a "plug-and-play" experience for the user.

## 🤖 The Multi-Agent Pipeline
The analysis is orchestrated via a 5-agent pipeline:
* **Chief Data Officer (CDO):** Defines high-level strategy and join maps (single-sheet and multi-sheet variants).
* **Senior Analyst:** Conducts deep synthesis and code-verified reasoning via the `execute_python_analysis` tool.
* **Self-Validation Pass:** A second Analyst call acting as an internal checklist for tautologies, noise correlations, and missing business impact — before the Critic sees the draft.
* **Critic Auditor:** Ruthless evaluator blocking structural failures (math errors, ID misuse, hallucinations).
* **Senior Presentation Architect:** Transforms the validated findings into a structured `PresentationSpec` for deterministic rendering.

## 🚀 Quick Start

### Prerequisites
* Python 3.12+
* An [OpenRouter](https://openrouter.ai/) API key

### Installation
```bash
git clone https://github.com/yourusername/agentic-data-analyst.git
cd agentic-data-analyst
uv sync
```

### Setup Environment
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_key_here
LOGFIRE_TOKEN=your_logfire_token_here
```

### Usage
Run an analysis mission via the CLI:
```bash
# Auto-routes based on sheet count (Single vs Multi-sheet)
python main.py data/sales_data.xlsx

# Force analysis on a specific sheet
python main.py data/complex_workbook.xlsx --sheet Q4_Report

# Analyze a flat CSV
python main.py data/churn.csv
```

## 🛠️ Tech Stack
* **Orchestration:** Pydantic-AI
* **Observability:** Logfire (Full Traceability)
* **Database:** aiosqlite (Async SQLite)
* **Data Engine:** Pandas, NumPy
* **Presentation:** python-pptx

## 🧠 Engineering Lessons Learned
1. From Notebooks to Production Architecture
The project started as a series of exploratory Jupyter Notebooks. The key challenge was transitioning to a Modular Multi-Agent System. By refactoring logic into discrete services (like service_layer.py and memory_manager.py), we achieved a strict Separation of Concerns, making the system testable and scalable beyond a single script.

2. Strategic Tool Convergence: The "One Tool" Philosophy
Rather than overwhelming the LLM with dozens of specialized tools, we opted for a Single-Tool Strategy via execute_python_analysis. This forced the Analyst agent to solve problems through verifiable code rather than relying on internal LLM knowledge, drastically reducing hallucinations and making the reasoning process transparent and auditable.

3. State Awareness via Schema Hashing
Relational data is volatile. By implementing Schema Fingerprinting (SHA-256), the system can identify if it has encountered a specific dataset structure before. This "State Awareness" allows the memory manager to inject historical context—such as previously successful joins or failed attempts—directly into the prompt, preventing the agent from repeating expensive analytical mistakes.

4. Handling Reasoning Model Constraints
During integration with OpenAI's reasoning models, we identified that standard sampling parameters (like temperature) are deprecated for reasoning tasks. The system now dynamically sanitizes configuration settings based on the selected model to ensure clean, error-free execution in production environments.

## 🗺️ Strategic Roadmap: Path to Production

To evolve this engine into a scalable SaaS product, the following architectural milestones are planned:

* **Production-Scale Task Queue:** Transitioning from synchronous execution to an **Async Job Queue (Celery/Redis)**. This will allow the system to handle datasets far exceeding the current 50MB safety guard by offloading heavy Pandas computations to dedicated workers.
* **CI/CD Evaluation Harness:** Implementing an automated testing suite that runs the agents against "Golden Datasets." This harness will track **Critic Consistency Scores** over time, ensuring that prompt engineering updates do not degrade analytical accuracy.
* **Cloud-Native Data Connectors:** Expanding beyond local files to direct API integrations with **Google Sheets, Notion, and BigQuery**, allowing for real-time analysis of live business data.
* **Human-in-the-Loop (HITL) Checkpoints:** Adding an optional step where users can approve the "Join Map" generated during the Discovery Phase before the Analyst agent begins expensive computation.