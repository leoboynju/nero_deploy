# Third-Party Components

`pyAgxArm` is vendored from:

```text
https://github.com/agilexrobotics/pyAgxArm.git
```

The checked revision is recorded in `pyAgxArm.VERSION`.

The ROS hardware packages under `ros_ws/src` are the minimal driver packages
needed by this deployment:

```text
agx_arm_ctrl
agx_arm_msgs
```

They are built locally by `scripts/install.sh` and are not loaded from a user
workspace at runtime.
