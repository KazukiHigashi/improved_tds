import importlib.util
import json
from pathlib import Path
import pytest


def launcher():
    path=Path(__file__).resolve().parents[1]/'scripts/run_scissors_retention_v4.py'
    spec=importlib.util.spec_from_file_location('retention_launcher_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_queue_completed_and_failed(tmp_path):
    module=launcher();messages=[]
    status=tmp_path/'status.json'
    status.write_text(json.dumps({'tasks':{'3':'completed','4':'completed','5':'completed'}}))
    module.wait_for_previous_run(tmp_path,messages.append,interval_seconds=0)
    assert len(messages)==1
    status.write_text(json.dumps({'tasks':{'3':'failed: exit 1'}}))
    with pytest.raises(RuntimeError,match='previous run failed'):
        module.wait_for_previous_run(tmp_path,messages.append,interval_seconds=0)


def test_queue_waits_for_all_tasks_without_agent_polling(tmp_path,monkeypatch):
    module=launcher();calls=[]
    status=tmp_path/'status.json'
    status.write_text(json.dumps({'tasks':{'3':'running','4':'completed'}}))
    def finish(seconds):
        calls.append(seconds)
        status.write_text(json.dumps({'tasks':{'3':'completed','4':'completed'}}))
    monkeypatch.setattr(module.time,'sleep',finish)
    module.wait_for_previous_run(tmp_path,lambda _:None)
    assert calls==[60]


def test_queue_missing_status_and_timeout_fail_closed(tmp_path):
    module=launcher()
    with pytest.raises(FileNotFoundError):module.wait_for_previous_run(tmp_path,lambda _:None)
    (tmp_path/'status.json').write_text('{')
    with pytest.raises(TimeoutError):module.wait_for_previous_run(tmp_path,lambda _:None,timeout_seconds=0)
