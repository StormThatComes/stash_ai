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
    import json
    import os
    import time
    from types import SimpleNamespace
    import pytest
    tk=pytest.importorskip('tkinter')
    try: root=tk.Tk()
    except tk.TclError:
        if os.environ.get('STASH_AI_DESKTOP_TEST'):raise
        pytest.skip('No graphical desktop')
    from stash_ai.recognition.performer_window import PerformerInspector
    from stash_ai.recognition.inspection_graph_window import GraphWindow
    from stash_ai.recognition.database import FaceDatabase
    config,person=scene_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        target, witness = db.execute("SELECT id,payload FROM candidates WHERE person=? AND status='accepted' "
                                     "ORDER BY rowid LIMIT 2", (person,)).fetchall()
        payload = json.loads(target['payload'])
        medoid = json.loads(witness['payload'])['observation']
        payload.update(reason='anchor_view_identity', anchor_view=dict(medoid=medoid, support_ids=[medoid]))
        db.execute('UPDATE candidates SET payload=? WHERE id=?', (json.dumps(payload), target['id']))
        db.commit()
    ui=PerformerInspector(root,IdentityReader(config,identity_id=person),'Test')
    def wait(predicate):
        deadline=time.monotonic()+8
        while not predicate() and time.monotonic()<deadline:
            root.update();time.sleep(.01)
        assert predicate(), ui.status.get()
    def text_items(graph):
        return {item: graph.canvas.itemcget(item, 'text') for item in graph.item_nodes
                if graph.canvas.type(item) == 'text'}
    try:
        wait(lambda:bool(ui.rows) and ui.summary_data is not None)
        assert ui.summary_data['observations']==7
        graph=GraphWindow(ui);ui.visual_windows.append(graph)
        wait(lambda:graph.graph is not None)
        assert graph.graph['total']==7
        assert not graph.loading and len(graph.thumbnail_images)==7
        assert graph.thumbnail_items  # Collapsed groups show labelled samples.
        view_group = next(n['id'] for n in graph.nodes.values() if n.get('label') == 'View matches')
        def group_labels():
            return {label for item, label in text_items(graph).items() if graph.item_nodes[item] == view_group}
        assert group_labels() == {'View ×1', 'Anchor ×1', 'View matches · sample'}
        images=dict(graph.thumbnail_images)
        graph.show_thumbnails.set(False);graph.rebuild()
        assert not graph.thumbnail_items
        assert group_labels() == {'View ×1', 'Anchor ×1', 'View matches'}
        graph.expand_all()
        assert {'View', 'Anchor', 'Auxiliary'} <= set(text_items(graph).values())
        graph.collapse_all()
        graph.show_thumbnails.set(True);graph.rebuild()
        assert graph.thumbnail_images==images
        graph.refresh()
        assert graph.loading
        graph.cancel_loading()
        assert not graph.loading and graph.thumbnail_images==images
        assert graph.graph['total']==7
        graph.expand_all();graph.pause();graph.fit()
        assert {'View', 'Anchor', 'Auxiliary'} <= set(text_items(graph).values())
        target = next(n for n in graph.nodes.values() if n.get('state') == 'anchor')
        graph.highlight = {target['id']}
        graph.draw()
        badge = next(item for item, label in text_items(graph).items()
                     if label == 'Anchor' and graph.item_nodes[item] == target['id'])
        graph.canvas.tag_raise(badge)  # Exercise double-clicking the label even if two nodes overlap.
        x, y = graph.canvas.coords(badge)
        graph.activate(SimpleNamespace(x=round(x), y=round(y)))
        wait(lambda:ui.current_row is not None and ui.current_row['id']==target['observation'] and ui.detail_data is not None)
        graph.collapse_all();graph.refresh()
        wait(lambda:not graph.loading and graph.worker.future is None)
        assert graph.thumbnail_items
        graph.refresh()  # Closing also cancels a load before any completion arrives.
        graph.close();graph.close()
        assert graph not in ui.visual_windows
    finally:
        ui.close();root.destroy();del ui,root;gc.collect()
