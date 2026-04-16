# mxQueryChat

**Ask questions about your data in plain language. Get answers as tables and charts.**

No need to know SQL. No need to know programming. Just type a question like *"Which cities sold the most tickets in 2025?"* and the app figures out the rest.

---

## What does this app do?

Imagine you have a giant spreadsheet with thousands of rows of ticket sales data. Normally, to get answers from it, you'd need to write special computer code called **SQL**. Most people don't know how to do that.

mxQueryChat solves that problem. You type a question in plain English (or German), and the app:

1. **Translates your question into SQL** (the special database language) using a local AI
2. **Shows you the SQL** so you can review it before anything runs
3. **Runs the query safely** — it can only *read* data, never change or delete it
4. **Shows you the results** as a table and a bar chart

Everything runs on your own computer. No data leaves your machine. No subscription fees.

---

## What data does it work with?

The app comes with a database of **German public transport ticket sales** — things like which ticket types were sold, how much revenue was made, and which regions bought the most tickets.

Example questions you can ask:
- *"Which 5 federal states generated the most ticket revenue in 2025?"*
- *"Show me monthly revenue for 2024 vs 2025"*
- *"What are the top 5 ticket types by revenue?"*

---

## Before you start: what you need

You need two things installed on your computer:

### 1. Python
Python is a free programming language. Think of it as the engine that runs this app.

- Download it from [python.org](https://www.python.org/downloads/)
- During installation, check the box that says **"Add Python to PATH"**
- Version 3.8 or newer works fine

### 2. Ollama
Ollama is a free program that runs an AI model on your computer (like having a mini ChatGPT that works offline).

- Download it from [ollama.com](https://ollama.com/)
- After installing, open a terminal and run:
  ```bash
  ollama pull qwen2.5-coder:1.5b
  ```
  This downloads the AI model (about 1 GB). You only need to do this once.

> **What is a terminal?**
> On Windows: press the Windows key, type "cmd" or "PowerShell", press Enter.
> On Mac: press Cmd+Space, type "Terminal", press Enter.

---

## Installation (one-time setup)

Open a terminal, navigate to the project folder, and run these commands one by one:

```bash
# Step 1: Create an isolated Python environment for this project
python -m venv .venv

# Step 2: Activate it
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Step 3: Install the required packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You only need to do this once. After that, skip straight to "Running the app."

---

## Running the app

Every time you want to use the app:

**Step 1** — Start Ollama (in one terminal window):
```bash
ollama serve
```
Leave this running in the background.

**Step 2** — Start the app (in a new terminal window, from the project folder):
```bash
# Activate the environment first (if not already active):
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

# Then start the app:
streamlit run app.py
```

**Step 3** — Open your browser and go to:
```
http://localhost:8501
```

The app will load and you can start asking questions!

---

## How to use the app

The app has three pages, accessible from the sidebar on the left:

### Ask a Question
This is the main page. Type your data question in the text box and press Enter. The app will:
- Generate SQL from your question
- Show you the SQL (you can click "Generate Explanation" if you want a plain-English explanation of what it does)
- Run it and display the results as a table and chart
- Let you give a thumbs up or thumbs down on the answer quality

### Chat History
See all your previous questions and answers from this session. Useful for going back to something you asked earlier.

### Training Data
This is where you can teach the app to answer questions better. You can add example question-and-SQL pairs. The more good examples you add, the smarter the app gets at answering your questions accurately.

---

## Configuration (optional)

Copy the file `.env.example` to a new file named `.env`. You can open it with any text editor (like Notepad) to adjust settings.

The most useful settings:

| Setting | What it does | Default |
|---|---|---|
| `OLLAMA_MODEL` | Which AI model to use for SQL generation | `qwen2.5-coder:1.5b` |
| `OLLAMA_URL` | Where Ollama is running | `http://127.0.0.1:11434` |
| `EXPLANATION_MODEL` | AI model used to explain SQL in plain English | `mistral` |
| `APP_LOG_LEVEL` | How detailed the logs are (`DEBUG`, `INFO`, `WARNING`) | `INFO` |

If the app feels slow, you can try a smaller Ollama model. If explanations time out, increase `EXPLANATION_TIMEOUT_SECONDS`.

---

## Safety guarantees

This app is designed to be safe with your data:

- **Read-only**: The app can only read data. It cannot insert, update, or delete anything. This is enforced in code and cannot be bypassed through questions.
- **Local-only**: Your data and questions never leave your machine. No internet connection is needed once everything is installed.
- **Complexity limits**: Overly complex or slow queries are automatically blocked before they run.
- **Query timeout**: Queries that take longer than 15 seconds are automatically cancelled.

---

## Project structure (for developers)

```
mxquerychat/
├── app.py                        # Main Streamlit app (the UI)
├── vannaagent.py                 # AI setup (Vanna + Ollama + ChromaDB)
├── sql_guard.py                  # Read-only SQL safety checks
├── mxquerychat.duckdb            # The local database file
├── src/
│   ├── core/query_logic.py       # Question-to-SQL resolution logic
│   ├── db/data_source.py         # Database metadata and caching
│   ├── db/execution_policy.py    # Complexity limits and timeouts
│   ├── llm/sql_explainer.py      # Plain-English SQL explanation
│   └── utils/telemetry.py        # Logging and metrics
├── training_data/
│   └── training_examples.csv     # Example Q&A pairs for training
├── tests/                        # Automated tests
├── tools/                        # Helper scripts (benchmarks, metrics)
├── docs/                         # Documentation and schema
├── .env.example                  # Configuration template
└── requirements.txt              # Python dependencies
```

### How a question gets answered

```
Your question
     |
     v
[Cache check] --> already answered? return instantly
     |
     v
[Training examples] --> exact match found? use that SQL
     |
     v
[Template planner] --> pattern recognized? build SQL deterministically
     |
     v
[AI (Ollama)] --> ask the local LLM to generate SQL
     |
     v
[Safety check] --> read-only? complexity OK? --> block if not
     |
     v
[Run query] --> execute on DuckDB with timeout
     |
     v
Results shown as table + chart
```

---

## Running tests

```bash
# Windows
.venv\Scripts\python -m pytest

# Mac/Linux
.venv/bin/python -m pytest
```

All 53 tests should pass.

---

## Viewing metrics

The app quietly tracks things like how many questions were asked and how many succeeded. To see a summary:

```bash
# Windows
.venv\Scripts\python tools/summarize_metrics.py

# Mac/Linux
.venv/bin/python tools/summarize_metrics.py
```

Output goes to `reports/metrics_summary.json`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot connect to Ollama" | Make sure `ollama serve` is running in a separate terminal |
| Explanation times out | Increase `EXPLANATION_TIMEOUT_SECONDS` in `.env`, or check Ollama is running |
| App won't start | Make sure your virtual environment is activated (`activate`) |
| Model not found | Run `ollama pull qwen2.5-coder:1.5b` to download it |
| Slow responses | The first question after startup is always slower (model loading). Subsequent ones are faster. |

---

## Tech stack (for the curious)

| Tool | Role |
|---|---|
| [Streamlit](https://streamlit.io/) | Web interface (the visual app in your browser) |
| [DuckDB](https://duckdb.org/) | Lightweight local database (like SQLite but faster for analytics) |
| [Vanna](https://vanna.ai/) | NL-to-SQL framework |
| [Ollama](https://ollama.com/) | Runs AI models locally on your computer |
| [ChromaDB](https://www.trychroma.com/) | Stores training examples as searchable vectors |
| [Pandas](https://pandas.pydata.org/) | Data manipulation and table display |

---

## Limitations

- Works with one database only (`mxquerychat.duckdb`)
- Supports German public transport data out of the box; other data requires setup
- AI-generated SQL can sometimes be wrong — always review before trusting results
- Not designed for production use; this is a learning/demo project
