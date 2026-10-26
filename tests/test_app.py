"""
Integration tests for Streamlit Causal Workbench.
Project 1: Neuro-Symbolic Causal Inference (Phase 1)
Team T094, Section E
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_every_page_renders_without_exception():
    """
    Verify that all 6 application pages load properly and that the
    interactive prediction button executes without raising uncaught exceptions.
    """
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception, f"Initial page render failed: {app.exception}"

    pages_to_test = [
        "Overview",
        "Reasoning lab",
        "Text → DAG",
        "Intervention lab",
        "Evaluation",
        "Team & viva",
    ]

    for target_page in pages_to_test:
        app.sidebar.radio[0].set_value(target_page).run()
        assert not app.exception, (
            f"Page '{target_page}' raised an exception: {[e.message for e in app.exception]}"
        )

    # Test interactive execution in Reasoning lab
    app.sidebar.radio[0].set_value("Reasoning lab").run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception, f"Evaluation button raised an error: {app.exception}"
