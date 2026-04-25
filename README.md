# Fake Social Media Account Detection System

This project is an **end-to-end fake social media account and content detection system** combining:

- **Sequential behavior models** (LSTM, Transformer)
- **Graph analysis** (GNN over interaction graphs)
- **NLP-based content analysis**
- **Cross-platform identity linking**
- A **clean, professional Streamlit dashboard** for interactive exploration.

## Features

- **Individual fake account detection**
- **Coordinated bot cluster discovery** via interaction graphs
- **Fake / AI-generated / spam content detection**
- **Lifecycle analysis** of accounts (creation → warm-up → attack → dormant)
- **Cross-platform identity linking** based on behavior, timing, and content style

The code is structured so you can plug in **real data and trained models**; by default it uses lightweight demo models to illustrate the full pipeline.

## Installation

```bash
cd path/to/project
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

> Note: `torch-geometric` and `sentence-transformers` are **optional** but recommended. If installation fails, you can comment them out in `requirements.txt` and the code will fall back to simpler implementations.

## Running the Streamlit Dashboard

```bash
streamlit run app.py
```

This will open a browser window with multiple pages:

- **Overview**: project summary and pipeline visualization
- **Single Account Analysis**: upload per-account CSV/JSON and run detection
- **Bot Cluster Detection**: build and visualize an interaction graph
- **Lifecycle Analysis**: estimate lifecycle stage of accounts
- **Cross-Platform Linking**: compare accounts across platforms

## Project Structure

- `app.py` – main Streamlit app
- `core/` – core models and fusion logic
- `features/` – feature extraction and preprocessing utilities
- `demo_data/` – small synthetic examples for quick testing

## Next Steps

- Replace demo models with **trained models** on your own datasets.
- Extend feature engineering in `features/` for your platform(s).
- Harden the pipeline (logging, monitoring, model versioning) for production.

