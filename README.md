# 🧠 Agentic Data Analyst: Autonomous Relational Data Engine

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Pydantic-AI](https://img.shields.io/badge/Framework-Pydantic--AI-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458)
![SQLite](https://img.shields.io/badge/Memory-aiosqlite-003B57)
![Logfire](https://img.shields.io/badge/Observability-Logfire-red)

An enterprise-grade, multi-agent AI system designed to autonomously profile, analyze, and visualize complex relational datasets (Excel/CSV). By separating deterministic data discovery from LLM reasoning, this engine eliminates common pitfalls of standard AI data tools—such as hallucinated joins or mathematical errors—and outputs executive-ready, natively editable PowerPoint presentations.

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

## 🤖 The Multi-Agent Pipeline
The analysis is orchestrated via a specialized 5-agent pipeline:
* **Chief Data Officer (CDO):** Defines high-level strategy and join maps.
* **Senior Analyst:** Conducts deep synthesis and code-verified reasoning.
* **Self-Validation Pass:** Internal checklist for tautologies and business impact.
* **Critic Auditor:** Ruthless evaluator blocking structural failures (math errors, ID misuse).
* **Senior Presentation Architect:** Transforms findings into a strategic slide deck.

## 🚀 Quick Start

### Prerequisites
* Python 3.12+
* An [OpenRouter](https://openrouter.ai/) API key

### Installation
```bash
git clone [https://github.com/yourusername/agentic-data-analyst.git](https://github.com/yourusername/agentic-data-analyst.git)
cd agentic-data-analyst
pip install -r requirements.txt
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