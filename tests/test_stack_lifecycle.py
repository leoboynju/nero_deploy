import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 5

# Every hardware-facing entry point is replaced, including recovery's Python.
# Gates let the tests control completion without relying on child sleep times.
CHILD = r"""
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(os.environ['TEST_STATE'])
role = sys.argv[1]
if role == '-m':
    assert sys.argv[2] == 'nero_deploy.diagnostics.recover'
    role = sys.argv[3]

def event(name, **extra):
    row = dict(role=role, event=name, pid=os.getpid(), **extra)
    fd = os.open(root / 'events', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.write(fd, (json.dumps(row) + '\n').encode())
    os.close(fd)

def gate(suffix):
    path = root / (role + suffix)
    while not path.exists():
        time.sleep(0.01)
    return path.read_text()

def terminate(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    event('term')
    gate('.term-release')
    event('exit', status=143)
    raise SystemExit(143)

signal.signal(signal.SIGTERM, terminate)
event('start', args=sys.argv[2:])
if role == 'drivers' and os.environ.get('DRIVERS_STATUS', '0') == '0':
    (root / 'driver-running').touch()
if role == 'stop':
    pidfile = root / 'client.pid'
    event('client-state', present=pidfile.exists() and
          Path('/proc', pidfile.read_text()).exists())
    (root / 'driver-running').unlink(missing_ok=True)
if role in ('client', 'left', 'right'):
    (root / (role + '.pid')).write_text(str(os.getpid()))
    status = int(gate('.exit'))
else:
    status = int(os.environ.get(role.upper() + '_STATUS', '0'))
event('exit', status=status)
raise SystemExit(status)
"""


class Stack:
    def __init__(self, root):
        self.root = root
        self.scripts = root / 'scripts'
        self.scripts.mkdir()
        self.processes = []
        self.env = {
            'PATH': '/usr/bin:/bin',
            'HOME': str(root),
            'TEST_STATE': str(root),
            'LOG_DIR': str(root / 'logs'),
            'KEEP_DRIVERS': '0',
        }
        for name in ('start.sh', 'recover_nero.sh'):
            shutil.copyfile(ROOT / 'scripts' / name, self.scripts / name)
        child = root / 'child.py'
        child.write_text(CHILD)
        command = f'exec {shlex.quote(sys.executable)} {shlex.quote(str(child))}'
        for name, role in (
            ('bind_can.sh', 'bind'),
            ('start_drivers.sh', 'drivers'),
            ('stop_all.sh', 'stop'),
            ('start_policy_client.sh', 'client'),
        ):
            (self.scripts / name).write_text(f'#!/bin/bash\n{command} {role} "$@"\n')
        python = root / 'fake-python'
        python.write_text(f'#!/bin/bash\n{command} "$@"\n')
        python.chmod(0o755)
        self.env['PYTHON'] = str(python)

    def start(self, script='start.sh', args=(), **env):
        process = subprocess.Popen(
            ['/bin/bash', str(self.scripts / script), *args],
            cwd=self.root, env={**self.env, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        self.processes.append(process)
        return process

    def events(self):
        path = self.root / 'events'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait_event(self, role, event):
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            for row in self.events():
                if (row['role'], row['event']) == (role, event):
                    return row
            time.sleep(0.01)
        pytest.fail(f'Missing {role}:{event}; events={self.events()}')

    def release(self, role, status=0, suffix='.exit'):
        path = self.root / (role + suffix)
        pending = path.with_name(path.name + '.pending')
        pending.write_text(str(status))
        pending.replace(path)

    def finish(self, process, status=0):
        stdout, stderr = process.communicate(timeout=TIMEOUT)
        assert process.returncode == status, (stdout, stderr, self.events())

    def assert_reaped(self, *roles):
        for row in self.events():
            if row['role'] in roles and row['event'] == 'start':
                assert not Path(f'/proc/{row["pid"]}').exists(), row


@pytest.fixture
def stack(tmp_path):
    stack = Stack(tmp_path)
    yield stack
    # Unblock synthetic children even when an assertion fails. Only signal the
    # private sessions created above, never a real deployment or driver.
    for role in ('client', 'left', 'right'):
        stack.release(role)
        stack.release(role, suffix='.term-release')
    for process in stack.processes:
        try:
            process.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=TIMEOUT)


@pytest.mark.parametrize('keep,status', [('0', 0), ('1', 0), ('0', 23), ('1', 23)])
def test_task_cleanup_and_keep_drivers_only_on_success(stack, keep, status):
    args = ('--task', 'synthetic task with spaces')
    process = stack.start(args=args, KEEP_DRIVERS=keep)
    assert stack.wait_event('client', 'start')['args'] == list(args)
    stack.release('client', status)
    stack.finish(process, status)
    retained = keep == '1' and status == 0
    assert (stack.root / 'driver-running').exists() == retained
    labels = [(row['role'], row['event']) for row in stack.events()]
    assert labels[:4] == [('bind', 'start'), ('bind', 'exit'),
                          ('drivers', 'start'), ('drivers', 'exit')]
    assert labels.count(('stop', 'start')) == (0 if retained else 1)
    if not retained:
        assert labels.index(('client', 'exit')) < labels.index(('stop', 'start'))
        assert not stack.wait_event('stop', 'client-state')['present']
    stack.assert_reaped('client')


@pytest.mark.parametrize('keep', ['0', '1'])
def test_preflight_failure_does_not_stop_recover_or_start_client(stack, keep):
    # An active controller may own these drivers. Launch-failure cleanup belongs
    # to start_drivers.sh, not the outer wrapper's preflight rejection path.
    driver = stack.root / 'driver-running'
    driver.write_text('existing active controller')
    process = stack.start(DRIVERS_STATUS='3', KEEP_DRIVERS=keep,
                          RECOVER_BEFORE_START='1')
    stack.finish(process, 3)
    assert [(row['role'], row['event']) for row in stack.events()] == [
        ('bind', 'start'), ('bind', 'exit'), ('drivers', 'start'), ('drivers', 'exit'),
    ]
    assert driver.read_text() == 'existing active controller'


def test_stack_lock_is_shared_across_log_directories(stack):
    first = stack.start(LOG_DIR=str(stack.root / 'first-logs'))
    stack.wait_event('client', 'start')
    second = stack.start(LOG_DIR=str(stack.root / 'second-logs'))
    stdout, stderr = second.communicate(timeout=TIMEOUT)
    assert second.returncode != 0, (stdout, stderr)
    assert 'already running' in stderr
    assert first.poll() is None
    assert [row['role'] for row in stack.events() if row['event'] == 'start'] == [
        'bind', 'drivers', 'client',
    ]
    stack.release('client')
    stack.finish(first)
    # A finished task must also release the shared lock.
    stack.finish(stack.start(LOG_DIR=str(stack.root / 'third-logs')))


def test_sigterm_forwards_and_waits_for_client_before_driver_cleanup(stack):
    process = stack.start(KEEP_DRIVERS='1')
    stack.wait_event('client', 'start')
    process.send_signal(signal.SIGTERM)
    stack.wait_event('client', 'term')
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=0.2)
    assert not any(row['role'] == 'stop' for row in stack.events())
    assert (stack.root / 'driver-running').exists()
    stack.release('client', suffix='.term-release')
    stack.finish(process, 143)
    labels = [(row['role'], row['event']) for row in stack.events()]
    assert labels.index(('client', 'term')) < labels.index(('client', 'exit'))
    assert labels.index(('client', 'exit')) < labels.index(('stop', 'start'))
    assert not stack.wait_event('stop', 'client-state')['present']
    assert not (stack.root / 'driver-running').exists()
    stack.assert_reaped('client')


@pytest.mark.parametrize('first', ['left', 'right'])
def test_recovery_waits_for_both_successful_children(stack, first):
    second = 'right' if first == 'left' else 'left'
    process = stack.start('recover_nero.sh')
    stack.wait_event('left', 'start')
    stack.wait_event('right', 'start')
    stack.release(first)
    stack.wait_event(first, 'exit')
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=0.2)
    assert not any(row['event'] == 'term' for row in stack.events())
    stack.release(second)
    stack.finish(process)
    assert not any(row['event'] == 'term' for row in stack.events())
    stack.assert_reaped('left', 'right')


@pytest.mark.parametrize('failed', ['left', 'right'])
def test_recovery_failure_cancels_and_reaps_other_child(stack, failed):
    other = 'right' if failed == 'left' else 'left'
    process = stack.start('recover_nero.sh')
    stack.wait_event('left', 'start')
    stack.wait_event('right', 'start')
    stack.release(failed, 17)
    stack.wait_event(failed, 'exit')
    stack.wait_event(other, 'term')
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=0.2)
    stack.release(other, suffix='.term-release')
    stack.finish(process, 17)
    stack.assert_reaped('left', 'right')


@pytest.mark.parametrize('first', ['left', 'right'])
def test_recovery_propagates_failure_after_other_child_succeeds(stack, first):
    second = 'right' if first == 'left' else 'left'
    process = stack.start('recover_nero.sh')
    stack.wait_event('left', 'start')
    stack.wait_event('right', 'start')
    stack.release(first)
    stack.wait_event(first, 'exit')
    with pytest.raises(subprocess.TimeoutExpired):
        process.wait(timeout=0.2)
    stack.release(second, 17)
    stack.finish(process, 17)
    assert not any(row['event'] == 'term' for row in stack.events())
    stack.assert_reaped('left', 'right')


@pytest.mark.parametrize('left,right', [(0, 0), (17, 0), (0, 17), (17, 17),
                                       (127, 0), (0, 127)])
@pytest.mark.parametrize('already_completed', [False, True])
def test_recovery_immediate_child_exits(stack, left, right, already_completed):
    for role, status in [('left', left), ('right', right)]:
        stack.release(role, status)
        stack.release(role, suffix='.term-release')
    env = {}
    if already_completed:
        # Let Bash reap both jobs before its real wait -n, without consuming
        # their saved exit statuses with a prior wait or mocking its result.
        harness = stack.root / 'wait-harness.bash'
        harness.write_text(r"""
wait() {
    if [[ "${1:-}" == '-n' ]]; then
        local pid result=0
        for pid in "${@:4}"; do
            while kill -0 "$pid" 2>/dev/null; do sleep 0.01; done
        done
        # Mark completed jobs as notified, keeping their saved wait statuses.
        jobs >/dev/null
        builtin wait "$@" || result=$?
        if [[ "$result" == 127 && -z "${completed:-}" ]]; then
            : > "$TEST_STATE/wait-n-exhausted"
        fi
        return "$result"
    fi
    builtin wait "$@"
}
""")
        env['BASH_ENV'] = str(harness)
    process = stack.start('recover_nero.sh', **env)
    stack.finish(process, left or right)
    if already_completed:
        assert (stack.root / 'wait-n-exhausted').exists()
        assert not any(row['event'] == 'term' for row in stack.events())
    stack.assert_reaped('left', 'right')
