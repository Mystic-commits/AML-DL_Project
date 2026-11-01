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
        "Viva demo",
        "Reasoning lab",
        "Text → DAG",
        "Intervention lab",
        "Evaluation",
        "Team & viva",
        "Extraction benchmark",
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


def test_guided_demo_requires_review_and_fits_real_network():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    app.sidebar.radio[0].set_value("Viva demo").run()
    assert not app.exception
    assert not app.button
    app.checkbox(key="demo_review").check().run()
    assert app.button[0].disabled
    app.checkbox(key="demo_assumptions").check().run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 5
    assert abs(float(app.metric[3].value) - 0.275) < 0.002
    app.radio(key="demo_state").set_value(1).run()
    assert not app.exception
    assert abs(float(app.metric[3].value) - 0.846) < 0.002
    app.checkbox(key="demo_review").uncheck().run()
    assert not app.metric
