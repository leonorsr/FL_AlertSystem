# FL Alert System

Simple Streamlit dashboard for fall-detection alerts while the federated backend
is still under development. It includes a user view and a caregiver view. The
technical origin of local data is not exposed in the interface.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`.

## Structure

- `dashboard/app.py`: Streamlit interface.
- `dashboard/mock_data.py`: simulated local data.

When the API is ready, the mock layer can be replaced with an HTTP client. Each
event should identify the local client, the monitored person and the caregiver
who receives the notification.
