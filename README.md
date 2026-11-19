# RideOps · Escalation Readiness Check (v0)

A Streamlit tool for L1 support agents. Before a ticket goes to engineering, it checks that the escalation won't come straight back. The case-study write-up is [`docs/writeup.pdf`](docs/writeup.pdf).

## Run it

You need Python 3.10 or newer.

```bash
cd rideops-escalation-check
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**. It loads `data/rideops_support_tickets.csv` automatically, so there's nothing to configure.

## Try it

1. Pick an agent in the sidebar ("Signed in as").
2. In **Escalate a ticket**, open a ticket from the queue (ones engineering sent back are listed first), or switch to **New ticket**.
3. Fill the checklist and click **Check readiness**. You get *Not ready* (with what's missing) or *Ready* (with an escalation note to copy or download).
4. **Escalation quality (weekly)** shows the KPIs. **How this works** explains the logic and limitations.

## Optional

```bash
python -m pytest -q                # run the tests
python analysis/run_analysis.py    # regenerate the analysis numbers and charts in docs/
```

## Notes

- Readiness checks are logged to `data/escalation_checks.csv`. Delete it to start with a clean log.
- To stop the app, press `Ctrl+C` in the terminal.
- More detail on the logic, project layout and limitations: [`docs/technical_notes.md`](docs/technical_notes.md).
