from pathlib import Path

from streamlit.testing.v1 import AppTest

from ndt.workspace import Workspace


def test_reviewer_demo_opens_without_upload_and_uses_offline_training_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    next(b for b in app.sidebar.button if b.label == "打开合成演示项目").click().run()
    assert not app.exception
    assert app.metric[0].value == "15"
    assert any("合成演示项目" in w.value for w in app.warning)
    assert len(Workspace("data/workbench").images()) == 0
    app.sidebar.selectbox[0].set_value("训练中心").run()
    assert not app.exception
    assert next(n for n in app.number_input if n.label == "训练轮数").value == 1
    assert next(s for s in app.selectbox if s.label == "初始化方式").value == "从零训练（无需下载）"
