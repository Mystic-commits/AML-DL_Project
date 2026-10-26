from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_every_page_renders():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    assert not app.exception
    for page in ['Reasoning lab','Text → DAG','Intervention lab','Evaluation','Team & viva']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page,[e.message for e in app.exception])
    app.sidebar.radio[0].set_value('Reasoning lab').run()
    app.button[0].click().run()
    assert not app.exception
