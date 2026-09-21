"""Exercise real driver methods with AST loading: no ROS or SDK imports/buses."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thirdparty/ros_ws/src/agx_arm_ctrl/agx_arm_ctrl/agx_arm_ctrl_single_node.py"
LAUNCH = ROOT / "thirdparty/ros_ws/src/agx_arm_ctrl/launch/start_single_agx_arm.launch.py"
JOINTS = [f"joint{i}" for i in range(1, 8)]


@pytest.fixture
def driver():
    clock = NS(now=1000.0)

    def sleep(seconds):
        clock.now += seconds

    events = []
    arms = []
    scenario = NS(version="1.11", feedback="fresh", delay=0.0, disabled_for=0.0, bad_joint=7)

    class Arm:
        joint_nums = 7

        def __init__(self, config):
            self.version = config.get("firmeware_version", "default")
            self.connected_at = None
            self.speed_error = False
            self.cached_mode_speed = 50
            self.motion_mode_speeds = []

        def connect(self):
            self.connected_at = clock.now
            events.append((self.version, "connect"))

        def disconnect(self):
            events.append((self.version, "disconnect"))

        def get_firmware(self):
            events.append((self.version, "firmware"))
            sleep(0.02)
            return {"software_version": scenario.version} if scenario.version else None

        def get_driver_states(self, index):
            events.append((self.version, "feedback", index))
            if clock.now < self.connected_at + scenario.delay:
                return None
            state = NS(timestamp=clock.now, hz=10.0,
                       msg=NS(foc_status=NS(driver_enable_status=True)))
            if clock.now < self.connected_at + scenario.disabled_for:
                state.msg.foc_status.driver_enable_status = False
            if index == scenario.bad_joint:
                if scenario.feedback == "missing":
                    return None
                if scenario.feedback == "disabled":
                    state.msg.foc_status.driver_enable_status = False
                if scenario.feedback == "preconnect":
                    state.timestamp = self.connected_at - 0.001
                if scenario.feedback == "stale":
                    state.timestamp = clock.now - 0.6
                if scenario.feedback == "future":
                    state.timestamp = clock.now + 0.1
                if scenario.feedback == "nan":
                    state.timestamp = float("nan")
                if scenario.feedback == "zero_hz":
                    state.hz = 0.0
            return state

        def get_joint_angles(self):
            return NS(msg=[0.2] * 7, hz=100, timestamp=clock.now)

        def get_joint_enable_status(self, index):
            events.append((self.version, "enable_status", index))
            return True

        def enable(self):
            events.append((self.version, "enable"))
            return True

        def set_speed_percent(self, value):
            events.append((self.version, "speed", value))
            if self.speed_error:
                raise RuntimeError("speed failed")
            self.cached_mode_speed = value

        def set_tcp_offset(self, value):
            events.append((self.version, "tcp", value))

        def __getattr__(self, name):
            # Record any unexpected control calls too, rather than hiding them.
            def call(*args, **kwargs):
                if name.startswith("move_"):
                    # SDK moves send the cached mode frame before the target.
                    self.motion_mode_speeds.append((name, self.cached_mode_speed))
                events.append((self.version, name, args, kwargs))
            return call

    def create_arm(config):
        arm = Arm(config)
        arms.append(arm)
        return arm

    namespace = {
        "Node": object, "math": math,
        "time": NS(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=sleep),
        "ArmModel": NS(NERO="nero", PIPER="piper"),
        "PiperFW": NS(DEFAULT="default", V183="v183", V188="v188"),
        "NeroFW": NS(V111="v111", V112="v112"),
        "AgxArmFactory": NS(create_arm=create_arm),
        "create_agx_arm_config": lambda **kw: {**kw, "joint_limits": dict.fromkeys(JOINTS)},
        "SetParametersResult": NS,
        "GRIPPER_JOINT_NAME": "gripper",
        "REVO2_HAND_JOINT_NAMES": [],
    }
    tree = ast.parse(SOURCE.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    node = namespace["AgxArmRosNode"].__new__(namespace["AgxArmRosNode"])
    parameters = {}
    node.declare_parameter = lambda key, default: parameters.setdefault(key, default)
    node.get_parameter = lambda key: NS(value=parameters[key])
    node.get_logger = lambda: Mock()
    node._declare_parameters()
    assert parameters["preserve_controller_state"] is False
    parameters.update(arm_type="nero", preserve_controller_state=True, auto_enable=False,
                      speed_percent=30, enable_timeout=0.15)
    node._load_parameters()
    node.gripper = None
    node.hand = None
    return NS(node=node, events=events, arms=arms, scenario=scenario, clock=clock,
              sleep=sleep, parameters=parameters)


def ready(driver):
    driver.node._init_agx_arm()
    driver.node.control_ready = True
    driver.events.clear()
    return driver.node


def joint_message(names=None, positions=None):
    return NS(name=JOINTS if names is None else names,
              position=[0.2] * 7 if positions is None else positions, effort=[])


@pytest.mark.parametrize("version,count", [("1.10", 1), ("1.11", 2), ("1.12", 2)])
def test_passive_init_queries_reconnects_verifies_without_control(driver, version, count):
    driver.scenario.version = version
    driver.scenario.delay = 0.04
    driver.node._init_agx_arm()
    assert len(driver.arms) == count
    assert driver.node.enable_flag
    assert not driver.node.control_ready
    assert driver.node._pending_speed_percent == 30
    assert {event[1] for event in driver.events} <= {"connect", "disconnect", "firmware", "feedback", "tcp"}
    final = driver.arms[-1]
    reads = [event[2] for event in driver.events if event[1] == "feedback"]
    assert reads == list(range(1, 8)) * (len(reads) // 7)
    assert len(reads) >= 14
    assert all(event[0] == final.version for event in driver.events if event[1] == "feedback")
    assert driver.clock.now >= final.connected_at + driver.scenario.delay


def test_incompatible_auto_enable_rejected_before_factory(driver):
    driver.node.auto_enable = True
    with pytest.raises(ValueError, match="auto_enable=false"):
        driver.node._init_agx_arm()
    assert driver.arms == []
    assert driver.events == []


def test_waits_for_fresh_enabled_status_without_sending_enable(driver):
    driver.scenario.disabled_for = 0.06
    driver.node._init_agx_arm()
    assert driver.clock.now >= driver.arms[-1].connected_at + 0.06
    assert driver.node.enable_flag
    assert {event[1] for event in driver.events} <= {"connect", "disconnect", "firmware", "feedback", "tcp"}


@pytest.mark.parametrize("field,value", [("enable_timeout", 0), ("enable_timeout", float("nan")),
                                         ("enable_timeout", float("inf")), ("speed_percent", 0)])
def test_invalid_passive_configuration_never_connects(driver, field, value):
    setattr(driver.node, field, value)
    with pytest.raises(ValueError):
        driver.node._init_agx_arm()
    assert driver.arms == []


@pytest.mark.parametrize("feedback", ["missing", "disabled", "preconnect", "stale", "future", "nan", "zero_hz"])
@pytest.mark.parametrize("bad_joint", [1, 7])
def test_unverified_joint_fails_and_disconnects_final_driver(driver, feedback, bad_joint):
    driver.scenario.feedback = feedback
    driver.scenario.bad_joint = bad_joint
    with pytest.raises(RuntimeError, match="fresh enabled feedback"):
        driver.node._init_agx_arm()
    assert driver.events[-1] == ("v111", "disconnect")
    assert not driver.node.enable_flag
    assert not driver.node.control_ready
    assert {event[1] for event in driver.events} <= {"connect", "disconnect", "firmware", "feedback"}


def test_firmware_failure_disconnects_without_control(driver):
    driver.scenario.version = None
    with pytest.raises(RuntimeError, match="firmware"):
        driver.node._init_agx_arm()
    assert driver.events[-1] == ("default", "disconnect")
    assert {event[1] for event in driver.events} == {"connect", "firmware", "disconnect"}


def test_final_reconnect_failure_disconnects_without_enabling(driver, monkeypatch):
    namespace = driver.node._init_agx_arm.__func__.__globals__
    factory = namespace["AgxArmFactory"].create_arm

    def create_arm(config):
        arm = factory(config)
        if arm.version == "v111":
            arm.connect = Mock(side_effect=RuntimeError("reconnect failed"))
        return arm

    monkeypatch.setattr(namespace["AgxArmFactory"], "create_arm", create_arm)
    with pytest.raises(RuntimeError, match="reconnect failed"):
        driver.node._init_agx_arm()
    assert driver.events[-1] == ("v111", "disconnect")
    assert not driver.node.enable_flag
    assert not driver.node.control_ready
    assert {event[1] for event in driver.events} == {"connect", "firmware", "disconnect"}


def test_postconnect_but_old_enable_frame_is_not_fresh(driver):
    driver.node.enable_timeout = 0.8
    driver.scenario.delay = 0.65
    driver.scenario.feedback = "stale"
    with pytest.raises(RuntimeError, match="fresh enabled feedback"):
        driver.node._init_agx_arm()
    assert driver.clock.now - driver.arms[-1].connected_at >= 0.8
    assert driver.events[-1] == ("v111", "disconnect")


def test_gripper_gate_and_parameter_update_do_not_consume_pending_speed(driver):
    node = ready(driver)
    node._joint_states_callback(joint_message(["gripper"], [0.1]))
    node._control_gate_callback(NS(data=False), NS())
    node._move_j_callback(joint_message())
    result = node._on_set_parameters([NS(name="speed_percent", value=20)])
    assert result.successful
    assert node._pending_speed_percent == 20
    assert driver.events == []
    node._control_gate_callback(NS(data=True), NS())
    assert driver.events == []
    node._move_j_callback(joint_message())
    assert [event[1] for event in driver.events] == ["speed", "move_j"]
    assert driver.events[0][2] == 20
    assert node._pending_speed_percent is None
    node._move_j_callback(joint_message())
    assert [event[1] for event in driver.events] == ["speed", "move_j", "move_j"]
    node._on_set_parameters([NS(name="speed_percent", value=15)])
    assert len(driver.events) == 3
    node._move_j_callback(joint_message())
    assert driver.events[-2][1:] == ("speed", 15)


@pytest.mark.parametrize("callback", ["_move_j_callback", "_move_js_callback", "_joint_states_callback"])
@pytest.mark.parametrize("message", [joint_message(JOINTS[:-1], [0.2] * 6),
                                     joint_message(positions=[0.2] * 6),
                                     joint_message(positions=[float("nan")] * 7),
                                     joint_message(positions=[float("inf")] * 7),
                                     joint_message(JOINTS[:-1] + ["joint1"])])
def test_invalid_joint_command_cannot_apply_speed(driver, callback, message):
    node = ready(driver)
    getattr(node, callback)(message)
    assert driver.events == []
    assert node._pending_speed_percent == 30


@pytest.mark.parametrize("callback,fast,motion", [("_move_j_callback", False, "move_j"),
                                                 ("_move_js_callback", False, "move_js"),
                                                 ("_joint_states_callback", False, "move_j"),
                                                 ("_joint_states_callback", True, "move_js")])
def test_actual_joint_dispatch_applies_speed_first(driver, callback, fast, motion):
    node = ready(driver)
    node.fast_mode = fast
    getattr(node, callback)(joint_message())
    assert [event[1] for event in driver.events] == ["speed", motion]
    assert driver.events[-1][2] == ([0.2] * 7,)


@pytest.mark.parametrize("motion", ["p", "l", "c"])
@pytest.mark.parametrize("valid", [True, False])
def test_cartesian_validation_precedes_deferred_speed(driver, motion, valid):
    node = ready(driver)
    node._create_pose_cmd = lambda pose: pose
    pose = [0.1] * 6 if valid else [float("nan")] * 6
    message = NS(pose=pose, poses=[[0.1] * 6, [0.1] * 6, pose])
    callback = getattr(node, f"_move_{motion}_callback")
    if valid:
        callback(message)
        assert [event[1] for event in driver.events] == ["speed", f"move_{motion}"]
    else:
        with pytest.raises(ValueError, match="Invalid arm motion target"):
            callback(message)
        assert driver.events == []
        assert node._pending_speed_percent == 30


@pytest.mark.parametrize("invalid", [None, "index", "nan", "length", "empty"])
def test_mit_batch_validation_precedes_any_dispatch(driver, invalid):
    node = ready(driver)
    msg = NS(joint_index=[1, 7], p_des=[0.2, 0.2], v_des=[0.0, 0.0],
             kp=[10.0, 10.0], kd=[0.8, 0.8], torque=[0.0, 0.0])
    if invalid == "index":
        msg.joint_index[-1] = 8
    elif invalid == "nan":
        msg.torque[-1] = float("nan")
    elif invalid == "length":
        msg.kp = []
    elif invalid == "empty":
        msg = NS(**{key: [] for key in vars(msg)})
    node._move_mit_callback(msg)
    if invalid:
        assert driver.events == []
        assert node._pending_speed_percent == 30
    else:
        assert [event[1] for event in driver.events] == ["speed", "move_mit", "move_mit"]


def test_speed_failure_does_not_send_target_or_clear_pending(driver):
    node = ready(driver)
    node.agx_arm.speed_error = True
    with pytest.raises(RuntimeError, match="speed failed"):
        node._move_j_callback(joint_message())
    assert [event[1] for event in driver.events] == ["speed"]
    assert node._pending_speed_percent == 30


@pytest.mark.parametrize("seamless,motion", [(True, "move_j"), (False, "move_js")])
def test_explicit_emergency_consumes_speed_before_implicit_sdk_mode(driver, seamless, motion):
    node = ready(driver)
    node.is_switch_seamlessly = seamless
    assert node.agx_arm.cached_mode_speed == 50
    assert node.agx_arm.motion_mode_speeds == []
    response = NS()
    assert node._emergency_stop_callback(NS(), response) is response
    assert [event[1] for event in driver.events] == ["speed", motion]
    assert driver.events[0][2] == 30
    assert driver.events[-1][2] == ([0.2] * 7,)
    assert node.agx_arm.motion_mode_speeds == [(motion, 30)]
    assert node._pending_speed_percent is None
    assert node.is_mit_mode == (not seamless)


def test_explicit_home_consumes_speed(driver):
    node = ready(driver)
    node._wait_motion_done = lambda: True
    node._move_home_callback(NS(), NS())
    assert [event[1] for event in driver.events] == ["speed", "move_j"]
    assert driver.events[-1][2] == ([0] * 7,)
    assert node._pending_speed_percent is None


@pytest.mark.parametrize("seamless", [True, False])
@pytest.mark.parametrize("positions", [[], [0.2] * 6, [0.2] * 8,
                                       [0.2] * 6 + [float("nan")],
                                       [0.2] * 6 + [float("inf")]])
def test_invalid_emergency_target_logs_failure_without_control(driver, seamless, positions):
    node = ready(driver)
    node.is_switch_seamlessly = seamless
    node.agx_arm.get_joint_angles = lambda: NS(msg=positions, hz=100)
    logger = Mock()
    node.get_logger = lambda: logger
    response = NS()
    assert node._emergency_stop_callback(NS(), response) is response
    logger.error.assert_called_once_with("Emergency stop failed: Invalid arm motion target")
    logger.info.assert_not_called()
    assert driver.events == []
    assert node.agx_arm.motion_mode_speeds == []
    assert node._pending_speed_percent == 30


def test_emergency_speed_failure_is_caught_without_sending_target(driver):
    node = ready(driver)
    node.agx_arm.speed_error = True
    logger = Mock()
    node.get_logger = lambda: logger
    response = NS()
    assert node._emergency_stop_callback(NS(), response) is response
    logger.error.assert_called_once_with("Emergency stop failed: speed failed")
    logger.info.assert_not_called()
    assert [event[1] for event in driver.events] == ["speed"]
    assert node.agx_arm.motion_mode_speeds == []
    assert node._pending_speed_percent == 30


@pytest.mark.parametrize("seamless,motion", [(True, "move_j"), (False, "move_js")])
def test_nonpreserving_emergency_retains_existing_speed_behavior(driver, seamless, motion):
    node = driver.node
    node.preserve_controller_state = False
    node._init_agx_arm()
    node.is_switch_seamlessly = seamless
    node._on_set_parameters([NS(name="speed_percent", value=25)])
    driver.events.clear()
    response = NS()
    assert node._emergency_stop_callback(NS(), response) is response
    assert [event[1] for event in driver.events] == [motion]
    assert driver.events[-1][2] == ([0.2] * 7,)
    assert node.agx_arm.motion_mode_speeds == [(motion, 25)]
    assert node._pending_speed_percent is None


def test_invalid_runtime_speed_keeps_pending_value(driver):
    node = ready(driver)
    result = node._on_set_parameters([NS(name="speed_percent", value=101)])
    assert not result.successful
    assert node._pending_speed_percent == 30
    assert driver.events == []


@pytest.mark.parametrize("auto_enable", [True, False])
def test_legacy_default_keeps_initial_and_runtime_speed_behavior(driver, auto_enable):
    node = driver.node
    node.preserve_controller_state = False
    node.auto_enable = auto_enable
    node._init_agx_arm()
    commands = [event[1] for event in driver.events]
    assert ("enable" in commands) == auto_enable
    assert "enable_status" in commands
    assert "speed" in commands
    assert "feedback" not in commands
    driver.events.clear()
    assert node._on_set_parameters([NS(name="speed_percent", value=25)]).successful
    assert driver.events == [("v111", "speed", 25)]
    assert node._pending_speed_percent is None


def test_launch_exposes_opt_in_and_passes_parameter_without_ros_imports():
    namespace = {
        "LaunchDescription": lambda actions: actions,
        "DeclareLaunchArgument": lambda name, **kw: NS(name=name, **kw),
        "LaunchConfiguration": lambda name: name,
        "Node": lambda **kw: NS(**kw),
    }
    tree = ast.parse(LAUNCH.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(LAUNCH), "exec"), namespace)
    actions = namespace["generate_launch_description"]()
    argument = next(item for item in actions if item.name == "preserve_controller_state")
    assert argument.default_value == "false"
    assert argument.choices == ["true", "false"]
    assert actions[-1].parameters[0]["preserve_controller_state"] == "preserve_controller_state"
