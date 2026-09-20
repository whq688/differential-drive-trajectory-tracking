# Differential-Drive Trajectory Tracking

This project builds a small but complete control pipeline for a differential-drive mobile robot.

## Roadmap

1. **Kinematic simulation**: model the robot and generate reference trajectories. ✅
2. **P / PI / PID trajectory tracking** under a persistent speed bias. ✅
3. **Time-varying LQR / LQI tracking** and controller comparison. ✅
4. ROS 2 simulation integration. *(next)*

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the open-loop simulation

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
python -m src.simulate_lqr_tracking
python -m src.simulate_lqr_speed_bias
python -m src.simulate_lqi_speed_bias
python -m src.compare_controllers
```

The final comparison script writes `outputs/controller_comparison.png`.

With the current parameters, the final distance errors under an 85% actual-speed
actuator are:

```text
P:   0.152 m
PID: 0.012 m
LQR: 0.021 m
LQI: 0.002 m
```

### What the experiments show

- **P control** reacts to the current along-track error, but a persistent
  actuator bias leaves a non-zero steady-state error.
- **PI control** integrates the signed along-track error to learn a lasting
  speed correction, substantially removing the steady-state error.
- **PID control** also considers the error-change rate. Its derivative term is
  low-pass filtered because raw numerical derivatives are sensitive to rapid
  fluctuations.
- **Time-varying LQR** linearizes the robot around the current reference pose
  and computes a feedback gain from state and control-cost matrices. With a
  persistent speed bias, ordinary LQR reduces but does not remove the
  steady-state error because it has no error memory.
- **LQI** augments LQR with the integral of the signed along-track error. It
  maintains a speed correction that compensates for a constant actuator bias,
  giving the smallest final error in this simulated scenario.

## Controller-comparison scenario

All four controllers use the same conditions:

- Circular reference path: radius `2.0 m`, angular speed `0.35 rad/s`.
- Initial robot pose: `(x, y, heading) = (2.5 m, -0.4 m, pi/2)`.
- Discrete simulation step: `0.02 s`.
- Persistent actuator bias: the robot receives only `85%` of the commanded
  linear speed.

This is a deliberately simple kinematic simulation. The LQI result shows the
benefit of integral action for a constant bias; it does not mean LQI is always
the best controller for every robot or disturbance.

## What the model represents

The state is `x = [x_position, y_position, heading]`. The inputs are linear velocity `v` and angular velocity `omega`:

```text
x_dot     = v cos(heading)
y_dot     = v sin(heading)
heading_dot = omega
```

This is the standard unicycle kinematic model used as a starting point for many differential-drive robots.
