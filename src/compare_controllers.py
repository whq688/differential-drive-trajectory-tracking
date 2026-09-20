"""Compare P, PID, LQR and LQI under the same persistent speed bias."""

from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.controllers import (
    LQITrackingController,
    LQRTrackingController,
    PITrackingController,
    PIDTrackingController,
)
from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


@dataclass
class SimulationResult:
    x: np.ndarray
    y: np.ndarray
    distance_error: np.ndarray


def simulate(controller_type: str, time: np.ndarray, reference: tuple, dt: float) -> SimulationResult:
    """Run one controller with an 85%-speed actuator."""
    ref_x, ref_y, ref_heading, ref_v, ref_omega = reference
    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    if controller_type == "P":
        controller = PITrackingController(
            heading_gain=2.5, distance_gain=0.8, integral_gain=0.0
        )
    elif controller_type == "PID":
        controller = PIDTrackingController(
            heading_gain=2.5,
            distance_gain=0.8,
            integral_gain=0.10,
            derivative_gain=0.15,
        )
    elif controller_type == "LQR":
        controller = LQRTrackingController()
    elif controller_type == "LQI":
        controller = LQITrackingController()
    else:
        raise ValueError(f"Unknown controller: {controller_type}")

    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)
    distance_error = np.empty_like(time)
    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y
        if controller_type in ("P", "PID"):
            commanded_v, angular_velocity, distance_error[index], *_ = controller.command(
                robot.state.x, robot.state.y, robot.state.heading,
                ref_x[index], ref_y[index], ref_heading[index], ref_v[index], ref_omega[index], dt,
            )
        else:
            commanded_v, angular_velocity, distance_error[index], *_ = controller.command(
                robot.state.x, robot.state.y, robot.state.heading,
                ref_x[index], ref_y[index], ref_heading[index], ref_v[index], ref_omega[index], dt,
            )
        robot.step(0.85 * commanded_v, angular_velocity, dt)
    return SimulationResult(actual_x, actual_y, distance_error)


def main() -> None:
    dt = 0.02
    time, ref_x, ref_y, ref_heading, ref_v, ref_omega = circle(
        radius=2.0, angular_speed=0.35, duration=24.0, dt=dt
    )
    reference = (ref_x, ref_y, ref_heading, ref_v, ref_omega)
    controller_names = ("P", "PID", "LQR", "LQI")
    results = {name: simulate(name, time, reference, dt) for name in controller_names}

    figure, (path_axis, error_axis) = plt.subplots(1, 2, figsize=(13, 5))
    path_axis.plot(ref_x, ref_y, "--", color="black", label="reference circle")
    colors = {
        "P": "tab:red",
        "PID": "tab:orange",
        "LQR": "tab:blue",
        "LQI": "tab:green",
    }
    for name in controller_names:
        path_axis.plot(results[name].x, results[name].y, color=colors[name], label=name)
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_title("Trajectory Comparison (85% Actual Speed)")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend()

    for name in controller_names:
        error_axis.plot(time, results[name].distance_error, color=colors[name], label=name)
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("distance error (m)")
    error_axis.set_title("Tracking-Error Comparison")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend()

    figure.tight_layout()
    output_path = Path("outputs/controller_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")
    for name in controller_names:
        print(f"{name} final distance error: {results[name].distance_error[-1]:.3f} m")


if __name__ == "__main__":
    main()
