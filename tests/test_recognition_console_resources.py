from types import SimpleNamespace

from stash_ai.recognition import live_resources as live


def test_disk_panel_reuses_volume_snapshot_without_traversal(tmp_path, monkeypatch):
    config = SimpleNamespace(database=tmp_path/'faces.db', data={'execution': {'disk_total_bytes': 5*2**30}})
    budget = SimpleNamespace(volumes={7: dict(used=2**30, requested=4*2**30, reserved=2*2**30, free=4*2**30)})
    view = live.ConsoleResources(config, budget=budget)
    monkeypatch.setattr(live.shutil, 'disk_usage', lambda *_: (_ for _ in ()).throw(AssertionError('extra filesystem query')))
    text = view.disk(10)
    assert 'used 1.00 GiB' in text and 'held 2.00 GiB' in text and 'free 4.00 GiB' in text
    budget.volumes = {}
    assert view.disk(11) == text


def test_process_sampler_measures_cpu_cores_without_summing_shared_memory(monkeypatch):
    now, cpu = [10.], [1.]
    monkeypatch.setattr(live, 'time', SimpleNamespace(monotonic=lambda: now[0], process_time=lambda: cpu[0]))
    monkeypatch.setattr(live, 'process_rss', lambda: 2**30)
    meter = live.ProcessMeter()
    assert meter.sample()['cores'] is None
    now[0] += 2
    cpu[0] += 6
    budget = SimpleNamespace(used=1024, limit=2048, free=4096)
    with live.monitor(meter), live.scratch_usage(budget):
        row = meter.sample()
    assert row['cores'] == 3 and row['rss'] == 2**30 and row['scratch_reserved'] == 1024
    assert meter.scratch is None
