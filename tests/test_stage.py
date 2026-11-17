# Edited for release.
"""leg/stage.py, against a fake nvidia-smi: it ends what its stage leaves behind, never touches anything else, and never
reads a failed or busy GPU query as an empty GPU.

    uv run --no-project --python 3.12 --with pytest pytest -q tests/test_stage.py
"""
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

STAGE = pathlib.Path(__file__).resolve().parent.parent / "leg" / "stage.py"
FAKE = """#!/bin/sh
# --query-compute-apps lists the live, non-zombie pids written to $FAKE_GPU; FAKE_FAIL makes the query fail
case "$*" in *query-compute-apps*) [ -n "$FAKE_FAIL" ] && exit 9
  [ -f "$FAKE_GPU" ] && for p in $(cat "$FAKE_GPU"); do
    [ "$(awk '{print $3}' /proc/$p/stat 2>/dev/null)" != Z ] && kill -0 "$p" 2>/dev/null && echo "$p"; done; exit 0;;
*) echo fake;; esac
"""

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and pidfds")


def gone(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] in ("Z", "X")
    except OSError:
        return True


@pytest.fixture
def env(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvidia-smi").write_text(FAKE)
    (tmp_path / "bin" / "nvidia-smi").chmod(0o755)
    return {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "FAKE_GPU": str(tmp_path / "gpu"),
            "SUSURRUS_GPU_WAIT_TRIES": "3"}


def run(env, tmp_path, minutes, code):
    return subprocess.run([sys.executable, str(STAGE), str(minutes), str(tmp_path / "log"), str(os.getpid()), "--", sys.executable, "-c",
                           code], env=env, capture_output=True, text=True, timeout=120)


def test_an_orphan_the_stage_leaves_is_ended(env, tmp_path):
    r = run(env, tmp_path, 1, "import subprocess,os; p=subprocess.Popen(['sleep','300']); "
                              "open(os.environ['FAKE_GPU'],'w').write(str(p.pid))")
    assert r.returncode == 0 and "ending 1 leftover" in r.stdout
    assert gone(int((tmp_path / "gpu").read_text()))


def test_an_outside_process_is_never_touched(env, tmp_path):
    outside = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        (tmp_path / "gpu").write_text(str(outside.pid))
        r = run(env, tmp_path, 1, "print('evaluated x')")
        assert r.returncode == 3 and outside.poll() is None and "could not confirm" in r.stdout
    finally:
        outside.kill()
        outside.wait()


def test_a_failed_query_is_not_an_empty_gpu(env, tmp_path):
    r = run({**env, "FAKE_FAIL": "1"}, tmp_path, 1, "print('teacher x')")
    assert r.returncode == 3


def test_past_its_limit(env, tmp_path):
    t0 = time.time()
    r = run(env, tmp_path, 0.02, "import os,time; open(os.environ['FAKE_GPU'],'w').write(str(os.getpid())); "
                                 "time.sleep(300)")
    assert r.returncode == 124 and gone(int((tmp_path / "gpu").read_text())) and time.time() - t0 < 30


def test_a_command_failure_keeps_its_code(env, tmp_path):
    assert run(env, tmp_path, 1, "import sys; sys.exit(7)").returncode == 7


def test_stopped_by_its_parent_it_still_ends_its_group(env, tmp_path):
    p = subprocess.Popen([sys.executable, str(STAGE), "5", str(tmp_path / "log"), str(os.getpid()), "--", sys.executable, "-c",
                          "import os,time,subprocess; s=subprocess.Popen(['sleep','300']); "
                          "open(os.environ['FAKE_GPU'],'w').write(f'{os.getpid()} {s.pid}'); time.sleep(300)"],
                         env=env, stdout=subprocess.PIPE, text=True)
    for _ in range(100):
        if (tmp_path / "gpu").exists() and len((tmp_path / "gpu").read_text().split()) == 2:
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 143
    assert all(gone(int(x)) for x in (tmp_path / "gpu").read_text().split())


def test_a_busy_gpu_is_3_whatever_the_command_returned(env, tmp_path):
    outside = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        (tmp_path / "gpu").write_text(str(outside.pid))
        assert run(env, tmp_path, 1, "import sys; sys.exit(7)").returncode == 3
    finally:
        outside.kill()
        outside.wait()


def test_a_second_interrupt_cannot_cut_cleanup_short(env, tmp_path):
    """Found before the run: an INT during cleanup once escaped it. Stop the helper, interrupt it again while it is ending
    a child that ignores TERM (so cleanup must wait out the TERM grace), and the child must still be gone."""
    p = subprocess.Popen([sys.executable, str(STAGE), "5", str(tmp_path / "log"), str(os.getpid()), "--", sys.executable, "-c",
                          "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                          "open(os.environ['FAKE_GPU'],'w').write(str(os.getpid())); time.sleep(300)"],
                         env=env, stdout=subprocess.PIPE, text=True)
    for _ in range(100):
        if (tmp_path / "gpu").exists() and (tmp_path / "gpu").read_text():
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGINT)
    time.sleep(2)                                       # now inside the 10 s TERM grace
    p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 143 and "SIGKILL" in out and gone(int((tmp_path / "gpu").read_text()))


def test_members_that_cannot_be_ended_are_a_failure(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage", STAGE)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    monkeypatch.setattr(stage, "end_group", lambda group, me: [12345])
    monkeypatch.setattr(stage, "gpu_free", lambda: True)
    assert stage.finish(1, None, 0) == 3


def test_it_refuses_to_start_for_a_parent_that_is_not_its_own(env, tmp_path):
    """Found before the run: a parent that died before the helper read its pid went unnoticed. The pid is now passed in."""
    r = subprocess.run([sys.executable, str(STAGE), "1", str(tmp_path / "log"), "1", "--", sys.executable, "-c",
                        "open('ran', 'w')"], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert r.returncode == 143 and not (tmp_path / "ran").exists()


def test_a_signal_error_leaves_a_survivor_and_exit_3(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage", STAGE)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    victim = subprocess.Popen(["sleep", "300"], start_new_session=True)     # its own group, led by itself
    try:
        def refuse(fd, sig):
            raise PermissionError(1, "Operation not permitted")
        monkeypatch.setattr(stage, "pidfd_signal", refuse)
        monkeypatch.setattr(stage.time, "sleep", lambda s: None)
        clock = iter(range(0, 10_000, 100))
        monkeypatch.setattr(stage.time, "time", lambda: next(clock))
        assert stage.end_group(victim.pid, -1) == [victim.pid] and victim.poll() is None
    finally:
        victim.kill()
        victim.wait()
