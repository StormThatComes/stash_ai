from stash_ai.recognition.cli import parser
from stash_ai.recognition.identity_inspection import IdentityReader
from stash_ai.recognition.performer_inspection import summary
from tests.test_recognition_identity_inspector import scene_fixture


def test_performer_summary_matches_current_accepted_pages(tmp_path):
    config, person = scene_fixture(tmp_path)
    before = config.database.read_bytes()
    reader = IdentityReader(config, performer_id='900')
    result = summary(reader)
    rows, _ = reader.page(accepted_only=True)
    assert result['observations'] == len(rows) == 7
    assert result['scenes'] == 2
    assert result['person']['id'] == person
    assert config.database.read_bytes() == before
    assert parser().parse_args(['inspect-performer','--performer-id','900','--offline']).offline


def test_desktop_performer_graph_navigation_and_shutdown(tmp_path):
    import gc
    import os
    import time
    import pytest
    tk=pytest.importorskip('tkinter')
    try: root=tk.Tk()
    except tk.TclError:
        if os.environ.get('STASH_AI_DESKTOP_TEST'):raise
        pytest.skip('No graphical desktop')
    from stash_ai.recognition.performer_window import PerformerInspector
    from stash_ai.recognition.inspection_graph_window import GraphWindow
    config,person=scene_fixture(tmp_path)
    ui=PerformerInspector(root,IdentityReader(config,identity_id=person),'Test')
    def wait(predicate):
        deadline=time.monotonic()+8
        while not predicate() and time.monotonic()<deadline:
            root.update();time.sleep(.01)
        assert predicate(), ui.status.get()
    try:
        wait(lambda:bool(ui.rows) and ui.summary_data is not None)
        assert ui.summary_data['observations']==7
        graph=GraphWindow(ui);ui.visual_windows.append(graph)
        wait(lambda:graph.graph is not None)
        assert graph.graph['total']==7
        graph.expand_all();graph.pause();graph.fit()
        target=next(n for n in graph.graph['nodes'] if n.get('current_member'))
        graph.inspect(target)
        wait(lambda:ui.current_row is not None and ui.current_row['id']==target['observation'] and ui.detail_data is not None)
        graph.collapse_all();graph.refresh()
        wait(lambda:graph.worker.future is None)
        graph.close();graph.close()
        assert graph not in ui.visual_windows
    finally:
        ui.close();root.destroy();del ui,root;gc.collect()
