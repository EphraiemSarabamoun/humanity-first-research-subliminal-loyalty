# Edited for release.
"""Run one stage of the leg so that everything it starts can be ended, and only that.

    python leg/stage.py <minutes> <log> <parent pid> -- <command...>

<parent pid> is the pid of the shell that starts the helper (run_leg.sh passes $$): the helper refuses to start the
command unless it is still that process's child once the parent-death signal is armed.

This helper makes itself the leader of a new process group and runs the command inside it. Everything the command
starts (a vLLM EngineCore included) joins the group. Because the helper stays alive until cleanup is over, the group's
id cannot be recycled meanwhile. After the command ends or runs past its limit, every other process still in the group
is signalled: TERM, then KILL for any that remain. Each one is pinned with a pidfd and re-checked to still be in the
group before the signal, so a recycled pid is never hit. Then the GPU must be confirmed empty by a query that succeeds.

A vLLM EngineCore that outlives its stage can keep a rented GPU busy for hours, so before the run (2026-11-03) the
cleanup was made provably bound to the stage's own processes.

Every wait is bounded and nothing can interrupt the cleanup: the first stop (TERM, INT, or the parent's death) makes
the helper ignore any further stop, then it cleans up exactly as after a normal end.

Exit: 3 when a process of the stage could not be ended or the GPU could not be confirmed free (whatever the command
returned; the leg must stop); otherwise the command's code, 124 when it ran past its limit, 143 when this helper was
told to stop (its parent, run_leg.sh, died or was stopped).
"""
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time

PR_SET_PDEATHSIG = 1
LIBC = ctypes.CDLL(None, use_errno=True)
SYS_PIDFD_SEND_SIGNAL, SYS_PIDFD_OPEN = 424, 434        # the same on x86_64 and arm64 (Linux 5.1 / 5.3)


def pidfd_open(pid):
    """A pidfd pinning pid; os.pidfd_open where this Python has it, else the syscall (some builds lack it)."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    fd = LIBC.syscall(SYS_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        e = ctypes.get_errno()
        raise ProcessLookupError(e, os.strerror(e)) if e == errno.ESRCH else OSError(e, os.strerror(e))
    return fd


def pidfd_signal(fd, sig):
    """Signal the process a pidfd pins (never whatever later reuses its pid)."""
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig)
    if LIBC.syscall(SYS_PIDFD_SEND_SIGNAL, ctypes.c_int(fd), ctypes.c_int(sig), None, ctypes.c_uint(0)) < 0:
        e = ctypes.get_errno()
        raise ProcessLookupError(e, os.strerror(e)) if e == errno.ESRCH else OSError(e, os.strerror(e))


def group_of(pid):
    """The process group of a live pid from /proc; None if it is gone or a zombie (exited, holding nothing, waiting to
    be reaped). Fields after the ')' that ends the command name are state, ppid, pgrp, ..."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    return None if fields[0] in ("Z", "X") else int(fields[2])


def members(group):
    """Every live process of the group."""
    return [int(d) for d in os.listdir("/proc") if d.isdigit() and group_of(int(d)) == group]


def end_group(group, me):
    """TERM, then KILL, every process of the group except this helper, each pinned by a pidfd and re-checked to be in
    the group before the signal. A process that cannot be pinned is never signalled. Returns the members still alive
    after both grace periods (empty when the group is clear)."""
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        targets = [p for p in members(group) if p != me]
        if not targets:
            return []
        print(f"stage: ending {len(targets)} leftover process(es) of group {group} with {sig.name}", flush=True)
        for pid in targets:
            try:
                fd = pidfd_open(pid)
            except OSError as e:                        # gone (ESRCH) or cannot be pinned: never signalled
                if not isinstance(e, ProcessLookupError):
                    print(f"stage: cannot pin pid {pid} ({e}); not signalling it", flush=True)
                continue
            try:
                if group_of(pid) == group:              # the pidfd now pins this exact process; is it still ours?
                    pidfd_signal(fd, sig)
            except ProcessLookupError:
                pass
            except OSError as e:                        # cannot signal it: it stays a survivor, so the exit is 3
                print(f"stage: cannot signal pid {pid} ({e})", flush=True)
            finally:
                os.close(fd)
        deadline = time.time() + grace
        while time.time() < deadline and [p for p in members(group) if p != me]:
            time.sleep(0.5)
    return [p for p in members(group) if p != me]


def gpu_free(tries=int(os.environ.get("SUSURRUS_GPU_WAIT_TRIES", "60")), pause=2.0, query_seconds=30):
    """True once a successful query lists no compute process on the GPU. A failed or hung query never counts as
    empty."""
    for _ in range(tries):
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=query_seconds)
            if r.returncode == 0 and not r.stdout.strip():
                return True
        except subprocess.TimeoutExpired:
            print("stage: the GPU query hung; retrying", flush=True)
        time.sleep(pause)
    return False


class Stop(Exception):
    pass


def ignore_stops():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def stop(signum, frame):
    """The first stop: ignore every later one (so cleanup cannot be interrupted), then unwind to the cleanup."""
    ignore_stops()
    raise Stop


def run(minutes, log, cmd, parent):
    """Lead a new process group and run the command in it; returns (child or None, code). Every stop that arrives
    before this returns becomes code 143; stops are ignored once it returns."""
    child = None
    try:
        os.setpgid(0, 0)
        if LIBC.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:      # told to stop when run_leg.sh dies
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != parent:                     # the parent died before the death signal was armed
            ignore_stops()
            raise Stop
        with open(log, "w") as out:
            child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
            try:
                rc = child.wait(timeout=minutes * 60)
            except subprocess.TimeoutExpired:
                print(f"stage: past its {minutes:g}-minute limit", flush=True)
                rc = 124
        ignore_stops()                                 # the last statement inside the try: no stop can escape it
    except Stop:
        rc = 143
    return child, rc


def finish(me, child, rc):
    """End what is left of the group, reap the command, confirm the GPU is free; the exit code (see the docstring)."""
    left = end_group(me, me)
    if child is not None:
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            left = sorted(set(left) | {child.pid})
    if left:
        print(f"stage: could not end {len(left)} process(es) of the stage {left} (command exit {rc})", flush=True)
        return 3
    if not gpu_free():
        print(f"stage: could not confirm the GPU is free (command exit {rc}); nothing of the stage is left, and "
              f"nothing else is touched", flush=True)
        return 3
    return rc


def main(argv):
    signal.signal(signal.SIGTERM, stop)                # first, so every later stop is caught
    signal.signal(signal.SIGINT, stop)
    minutes, log, parent, cmd = float(argv[0]), argv[1], int(argv[2]), argv[argv.index("--") + 1:]
    child, rc = run(minutes, log, cmd, parent)
    return finish(os.getpid(), child, rc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
