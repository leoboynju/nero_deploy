# Nero Deployment

Self-contained robot-side deployment for the Nero three-camera, dual-arm
policy. The policy server is remote and is accessed with the OpenPI WebSocket
protocol.

## Layout

```text
config/       device and policy configuration
src/          deployment Python package
scripts/      install, device, driver, test, and control entry points
vendor/       vendored openpi-client protocol implementation
thirdparty/   vendored hardware SDKs
tests/        automated tests
outputs/      logs and transport captures
```

## Install

Install ROS2 Humble, `colcon`, and the normal ROS build dependencies first.
Then run:

```bash
bash scripts/install.sh
```

The script creates `.venv`, installs this project and `thirdparty/pyAgxArm`,
uses the vendored `openpi-client` directly, then builds `agx_arm_ctrl` and
`agx_arm_msgs` packages under `thirdparty/ros_ws`.

## Configure Devices

Edit `config/nero.yaml` or override values with environment variables. The
current configured mapping is:

```text
left wrist:   261822075208
right wrist:  261622075978
third person: 261722074322
left Nero:    can3
right Nero:   can2
```

Check devices and configure CAN:

```bash
bash scripts/check_devices.sh
bash scripts/bind_can.sh
```

`pyAgxArm`, `agx_arm_ctrl`, and `agx_arm_msgs` are all vendored under
`thirdparty`. No user-specific Python or ROS workspace is used by default.

## Check Camera Views

Capture the configured three cameras without starting ROS, connecting to the
policy server, or sending arm commands:

```bash
bash scripts/capture_camera_views.sh
```

Stop any other camera capture or RealSense Viewer session first. The script
uses the serial numbers, resolution, FPS, and exposure settings from
`config/nero.yaml`, warms up the cameras, and waits for fresh frames. Each run
creates a new timestamped directory under `outputs/camera_views/` containing
three original RGB PNGs and a labeled `overview.png`. The overview order is
left wrist, right wrist, third person; verify that the actual scene matches
these configured roles. Frames are not hardware-synchronized.

The Bash entry point runs `src/tests/capture_camera_views.py` using the project
virtual environment. It supports `PYTHON`, `CONFIG`, and `OUTPUT_DIR`
environment overrides and forwards all command-line arguments. Use
`--config PATH`, `--output-dir PATH`, or `--timeout SECONDS` to override the
configuration, output parent directory, or frame wait timeout after warmup.

## Test Policy Transport

This does not publish robot commands. It captures all three images, sends the
Nero observation format, verifies the returned action shape, and writes PNG,
NPY, and a manifest under `outputs/transport`:

```bash
bash scripts/test_policy_transport.sh
```

## Run

The Nero server now defaults to **5 flow denoising iterations** (both RTC and
ordinary inference). This is independent of the 16 predicted actions and the
client's default eight executed actions per synchronous request. Restart the
model server using `scripts/serve_pi05_nero.sh`; `--num-steps N` or `NUM_STEPS=N`
overrides its default. The current default checkpoint is `pi05_nero_new/10000`,
whose assets ID is `nero_datasets_lerobot`.

Choose one of the two explicit entry points. Both use the same `config/nero.yaml`
for the policy server, task prompt, cameras, arms and command filtering.

**RTC: background inference with prefix guidance**

```bash
bash scripts/start_rtc.sh
```

**No RTC: ordinary synchronous inference**

```bash
bash scripts/start_no_rtc.sh
```

The no-RTC entry point makes a blocking prediction, then executes the first eight
actions (`0..7`) of the 16-step chunk before requesting another prediction. No
RTC prefix or delay fields are sent. There can be a pause between chunks while
the server predicts. Choose a different number of executed actions with:

```bash
REPLAN_STEPS=5 bash scripts/start_no_rtc.sh   # First 5 actions, as in the LIBERO example's scheduling
REPLAN_STEPS=16 bash scripts/start_no_rtc.sh  # Execute the complete chunk
```

`REPLAN_STEPS` must be between 1 and 16. `--replan-steps N` also works. Both scripts
accept the existing `CONFIG`, `PROMPT`, `MAX_STEPS`, and `PYTHON` overrides and pass
client arguments through. The same running OpenPI policy server serves both modes.
For example:

```bash
MAX_STEPS=500 bash scripts/start_rtc.sh
MAX_STEPS=500 REPLAN_STEPS=8 bash scripts/start_no_rtc.sh
```

The scripts include CAN/driver startup and cleanup, just like `start.sh`. If the
drivers are already managed in another terminal, start only the policy client:

```bash
bash scripts/start_policy_client.sh --inference-mode rtc
bash scripts/start_policy_client.sh --inference-mode sync --replan-steps 8
```

The existing config-driven entry point remains available:

```bash
PROMPT="handover block with both arms" \
MAX_STEPS=2000 \
bash scripts/start.sh
```

For separate terminals, use `scripts/start_drivers.sh` followed by
`scripts/start_policy_client.sh`.

The client publishes:

```text
/left_arm/control/move_j
/left_arm/control/joint_states
/right_arm/control/move_j
/right_arm/control/joint_states
```

In `rtc` and `async` modes the control loop uses latest-chunk execution: inference runs in a single
background worker, and each newly received action chunk atomically replaces
the unexecuted part of the previous chunk. When `policy.rtc.enabled` is true,
the remaining previous actions are also sent with the Nero RTC fields. RTC
errors are propagated rather than silently switching algorithms. To compare
ordinary asynchronous inference, explicitly set `policy.rtc.enabled: false`.
This legacy setting selects **unguided async**, not synchronous execution. The
two named entry points explicitly override it. For a controlled comparison of
guidance alone, use `scripts/start_policy_client.sh --inference-mode async`.

It consumes:

```text
/left_arm/feedback/joint_states
/right_arm/feedback/joint_states
```

Images are captured at `640x480` and resized with padding to `224x224` on the
robot before WebSocket transport, matching the OpenPI remote-inference
recommendation. Set `policy.image_size` to `null` only when the server
explicitly requires the original image dimensions.

RTC execution uses `openpi_client.AsyncRTCActionChunkBroker`. The ROS loop
calls `broker.infer(observation)` once per control tick and receives one
`(16,)` action. The broker sends the first request without RTC fields, starts
background inference when the local queue reaches `trigger_horizon`, and sends
the unexecuted previous actions for later requests. It estimates inference
delay from measured round-trip latency. If `policy.rtc.enabled` is false, the
same broker uses ordinary inference requests and keeps the same local queue
behavior.

`--inference-mode` accepts `rtc`, `sync`, or `async`. Without the flag, an optional
`policy.inference_mode` config value is used, otherwise the legacy `policy.rtc.enabled`
selection applies. Direct sync CLI use defaults to `policy.sync_replan_steps` or
eight when that setting is absent. `control.max_episode_seconds: 0` disables the
wall-time limit, so synchronous waiting does not prematurely stop `max_steps` runs.
Action traces include the effective inference mode and sync replan length. The
shared `rtc_chunk_id`/`rtc_action_index` diagnostic names are also used for sync
traces; they do not indicate that RTC conditioning was sent to the server.

Each observation and its first previous action refer to the same control tick:
the broker submits the request before removing the action it is about to return.
It skips only the number of actions actually returned since that request began,
not `latency * control_hz`. Initial blocking inference starts at action zero, and
waiting on an empty queue does not advance the cursor. Use exactly one `infer`
call per published action, with a fresh observation before that call.

The request's `rtc_execution_horizon` is the actual remaining prefix length, so
an earlier `trigger_horizon` also extends the overlap used by the server. Timing
logs separate round-trip/polling latency, the conservative guidance delay estimate,
and `discard_steps` (the actual action-index offset). The ROS loop includes its
processing time in the control-period budget and avoids catch-up command bursts
after a blocked inference.

Start the updated OpenPI server with warmup enabled: full-Jacobian RTC compiles a
gradient-enabled denoiser path on its first call. After changing either side's
Python files, restart the policy server and robot policy client to load the changes.

RTC regression tests (no ROS control or cameras):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src:$PWD/vendor/openpi-client/src" \
  .venv/bin/python -m pytest tests/test_rtc_broker.py tests/test_rtc_regression.py tests/test_policy_client_rtc.py -q
```

This project does not reference a user's OpenPI, TipTop, or Pika workspace.

### Driver lifecycle and duplicate subscribers

Startup probes DDS directly (not the cached `ros2` daemon), requiring live
feedback from both arms and exactly one subscriber on each control topic. Missing
feedback is not interpreted as proof that a driver process has exited.

Drivers launch in separate process sessions. PID records include process start
time; cleanup stops their groups and identifies orphan driver nodes and launchers by this
deployment's installation path, namespace, user and ROS domain. It does not
kill arbitrary ROS processes or blindly follow reused PID files. Stale endpoints
must disappear before replacement drivers launch. Existing command publishers
block startup rather than having their drivers restarted underneath them.
If live drivers survive project cleanup, startup reports the conflict early instead
of waiting out the 25-second stale-endpoint timeout. A local read-only diagnostic
prints driver PIDs, installation paths, and launcher parents; remote DDS processes
may not be visible locally. Unknown node names are not ignored as stale cache.

Both mode wrappers now defer recovery until after the stack lock, controller
checks, driver cleanup, and an empty-graph check. CAN binding runs once per full
startup. Recovery opens both grippers and moves both arms; both recovery children
must finish successfully before driver launch. Failure of either cancels and reaps
the other. Package resolution is checked before recovery and launch to reject an
inherited external ROS overlay instead of creating untracked external drivers.

Recovery completion now requires fresh post-command status and all four joint
feedback groups, seven measured joints within 0.01 rad of the target, and at least
0.5 seconds of advancing samples with at most 0.002 rad of joint variation. A cached
`motion_status=0` alone no longer counts as arrival. Every stream must be at most
0.25 seconds old, with at most 0.1 seconds between stream timestamps. These are
software verification thresholds, not hardware-certified safety limits. Failure
blocks driver launch; disconnecting the SDK does not cancel a controller trajectory
or confirm that the arm stopped. Gripper arrival is not verified by this check.
The collector retains received status transitions and joint extrema between polls,
so a received excursion followed by a return to target cannot hide within a poll.
This still verifies delivered feedback, not continuous physical motion or all CAN
faults: the SDK drops CAN error frames before these callbacks and can clear transient
communication errors before polling. A timeout/fault must not be worked around by
relaxing the acceptance thresholds without diagnosing the hardware/feedback cause.

Project driver launches use `preserve_controller_state=true` and `auto_enable=false`.
After connecting and identifying firmware, they only verify fresh enabled feedback
from every joint; they do not re-enable the arm or write speed/mode/target commands
during initialization. `SPEED_PERCENT` now applies to both recovery and the next
accepted arm-motion command (default 30), rather than resetting speed to 100 at
driver startup. Speed parameter updates are deferred until an accepted arm-motion
command too. Disabled or unverified arms fail startup rather than being automatically
enabled. This is not a global safety gate: explicit external motion/enable/services
retain their own semantics. Keep other controllers stopped.

To clean up project-owned driver leftovers after an interrupted run:

```bash
bash scripts/stop_all.sh
```

This command only stops project drivers, not a running policy client or another
workspace's teleoperation stack. Stop the owning task/launch first. For a read-only
local process report in the same ROS domain:

```bash
python3 scripts/driver_processes.py diagnose
```

The startup and cleanup scripts serialize driver operations; the full `start.sh`
stack also has a run lock. Locks live under `outputs/locks`, independently of
`LOG_DIR`. Stop any running old-version stack before updating/restarting: it may
still hold a legacy lock under its log directory. Driver subprocesses do not inherit those lock file
descriptors. The policy client's exclusive-control check remains enabled.

### Repeated tasks

Default behavior remains recovery before each task and driver cleanup on exit.
For consecutive tasks, explicitly retain drivers after a successful task:

```bash
KEEP_DRIVERS=1 bash scripts/start_rtc.sh
```

If the robot's current pose and gripper state are suitable for the next task, skip
recovery and reuse healthy drivers on subsequent runs:

```bash
RECOVER_BEFORE_START=0 KEEP_DRIVERS=1 bash scripts/start_rtc.sh
```

The same options work with `start_no_rtc.sh`. Skipping recovery does not return the
arms to their initial pose or open the grippers. Live feedback, unique endpoints,
and exclusive control are still checked; cameras and policy connections are still
initialized each task. The configured camera exposure warmup is unchanged.
Do not use this option to bypass an external controller conflict.
Newly created drivers also require fresh evidence that the arms are already enabled;
skipping recovery is not an automatic hardware-enabling path. After any unexpected
motion, do not use fast restart as a diagnostic workaround or treat it as safe to retry.

Interrupts and task failures still clean up drivers, even with `KEEP_DRIVERS=1`.
The wrapper forwards termination to the policy client and waits for it to exit
before driver cleanup. If it remains blocked, it is force-terminated after 10
seconds; trace saving is not guaranteed in that fallback. Direct standalone client
use still has blocking network connection/receive calls. ROS cleanup is idempotent
and partial camera initialization is cleaned up. Stopping software is not an
emergency stop or a motor-disable command.

The client logs `no policy commands published yet` on connection and
`publishing first policy action` immediately before its first ROS control publication.
Movement before that publication must be investigated on the recovery/controller
handoff or another command source, not assumed to be a policy inference action.

### RTC execution indices and jitter traces

#### Joints-only RTC and coordinated gripper execution

The updated Nero server guides only action dimensions **0..6 and 8..14**.
Grippers (7, 15) and padding are excluded from both the prefix residual and the
VJP correction. Network coupling can still indirectly affect gripper predictions;
this is not a separate unguided gripper inference branch. The client checks the
server's `rtc_guided_action_indices` response and reports an explicit error if an
old server is still running. **Restart the updated model server and policy client.**

`control.grasp.enabled: true` applies the following coordination **only in RTC mode**.
All inference modes use the same asymmetric intent rule: **opening needs two
consecutive policy action samples; closing needs only one**. Left and right
counters are independent, and a close request clears any pending open count.
`action[7] >= 0.5` requests left opening and `action[15] >= 0.5` requests right
opening; lower values request closing. These are absolute goals, not increments.
In no-RTC modes the first close request is sent on that control tick, whereas
opening is sent on the second consecutive open request.
Original pre-inference joint feedback for EMA/clamping and normal queue execution
are retained. Shared grasp settings cannot activate coordination in those modes.
`start_no_rtc.sh` also explicitly passes `--no-grasp-control`.
Startup prints `Gripper execution: open2_close1` (or `coordinated_open2_close1` for
RTC coordination), and traces record `gripper_execution`.
The bridge maps open/close to widths 0.1/0.0 m; observation state is feedback,
not the source of the next gripper command.

1. Confirm opening on two consecutive new policy actions; accept closing on its
   first sample. Held/cached control ticks do not count as new confirmations.
2. For closure, remember the pose at the first close intent, then hold both arms; wait for the
   affected arm joints to stay within 0.025 rad for 60 ms. Subsequent lift/retract
   targets are not consumed. An unreachable pose aborts closure after 1.5 s.
3. Send the gripper command and hold the pose for at least 300 ms. Wait for reported
   closure, or width progress followed by stable width and contact effort (when
   effort is available). Effort alone during motion is not completion. Nonzero
   contact width is valid; a closed-empty gripper is not proof of a successful grasp.
   Release also has a bounded hold/feedback phase. The feedback timeout is 1 s.
4. On completion or timeout, discard queued/in-flight predictions and infer again
   from fresh observations. A timeout is logged, not reported as grasp success.

The timing, joint tolerance, width and force thresholds are configurable under
`control.grasp`. They are initial calibration values, not measured guarantees for
all objects. `close_timeout_seconds` currently bounds both opening and closing.
Missing effort stays NaN; width settling remains available. Feedback is the latest
ROS JointState data, not a hardware-synchronized contact measurement.

The RTC entry point enables this by default. To isolate the joints-only model
change from execution coordination, use the explicit ablation flag:

```bash
bash scripts/start_rtc.sh --no-grasp-control
```

That flag retains open-two/close-one confirmation without pose coordination; it does not re-enable RTC on the
gripper. Trace version 2 records `rtc_guidance_mask`, `grasp_phase`, `grasp_event`,
`gripper_force`, `grasp_target` and `action_advanced`. Held ticks still count as published control
steps toward MAX_STEPS, but do not consume the policy action queue. The analyzer
excludes held indices from skipped/repeated-action checks. `state` is refreshed
after inference only when RTC grasp coordination is active. No-RTC feedback timing
remains the original baseline.

RTC executes `new_chunk[actual_delay:]`, not a fixed final eight steps. For
16-step predictions with a request interval of eight and latency of five ticks,
steady-state chunks normally execute indices **5..12** (zero-based). With varying
latency, the executed count may be seven, eight, or nine. `RTC executed` and
`RTC adopt` logs report each chunk's exact execution range and origin/adoption ticks.
`num_steps=10` on the model is the flow solver iteration count, not the robot's
action execution count.

The official OpenPI LIBERO example is synchronous and runs the **first five**
predicted actions; it has no asynchronous RTC queue. LeRobot's RTC action queue
does use the suffix after the actual delay. Do not copy synchronous LIBERO's
front-of-chunk slicing into this controller, or skip directly to index eight.

`control.action_trace` saves each executed step's feedback, post-broker policy
target, EMA target, actual published command, chunk ID, and action index under
`outputs/action_traces/`. Gripper values in `command` are the published widths,
whereas gripper values in `policy_action` are model outputs. Traces are buffered
in memory and saved on exit. Analyze a run with:

```bash
.venv/bin/python scripts/analyze_action_trace.py outputs/action_traces/<timestamp>.npz
```

The summary separates same-timestamp plan changes, within/between-chunk target
steps, actual command steps, and feedback tracking error. Large plan changes at
switches point to cross-chunk inconsistency; smooth commands but oscillating
feedback point toward the lower-level control/tracking chain.

EMA starts from measured joints and commits the command after downstream limiting,
so its internal state does not drift toward an unsent target. The existing
The intent rule is fixed at open=2, close=1 for all modes. Legacy
`gripper_confirm_steps` and `grasp.confirm_seconds` values do not change it;
`confirm_seconds` is accepted only for compatibility with existing configs.

## Recovery

The recovery command uses the vendored `pyAgxArm` SDK directly:

```bash
bash scripts/recover_nero.sh
```

This immediately recovers both arms. Recovery defaults to 30% speed. Override it
when needed:

```bash
SPEED_PERCENT=50 bash scripts/recover_nero.sh
```
