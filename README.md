# Differential-Drive Trajectory Tracking

This project builds a small but complete control pipeline for a differential-drive mobile robot.

## Roadmap

1. **Kinematic simulation** (current): model the robot and generate reference trajectories.
2. PID trajectory tracking.
3. LQR trajectory tracking and disturbance comparison.
4. ROS 2 simulation integration.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the first simulation

```powershell
python -m src.simulate_open_loop
```

The script writes a plot to `outputs/open_loop_trajectory.png`.

## Feedback-control experiments

All comparison experiments start from a position offset and model an actuator
whose actual forward speed is only 85% of its commanded speed.

```powershell
python -m src.simulate_speed_bias
python -m src.simulate_pi_speed_bias
python -m src.simulate_pid_speed_bias
python -m src.compare_controllers
```

The comparison script writes `outputs/controller_comparison.png`.

With the current parameters, the final distance errors are:

```text
P:   0.152 m
PI:  0.015 m
PID: 0.012 m
```

### What the experiments show

- **P control** reacts to the current along-track error, but a persistent
  actuator bias leaves a non-zero steady-state error.
- **PI control** integrates the signed along-track error to learn a lasting
  speed correction, substantially removing the steady-state error.
- **PID control** also considers the error-change rate. Its derivative term is
  low-pass filtered because raw numerical derivatives are sensitive to rapid
  fluctuations.

## What the model represents

The state is `x = [x_position, y_position, heading]`. The inputs are linear velocity `v` and angular velocity `omega`:

```text
x_dot     = v cos(heading)
y_dot     = v sin(heading)
heading_dot = omega
```

This is the standard unicycle kinematic model used as a starting point for many differential-drive robots.
