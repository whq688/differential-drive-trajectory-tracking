"""Compare how different LQR state and control costs change tracking behaviour."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.controllers import LQRTrackingController
from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


@dataclass
class SimulationResult:
    x: np.ndarray
    y: np.ndarray
    distance_error: np.ndarray
    commanded_v: np.ndarray
    commanded_omega: np.ndarray


def simulate(time: np.ndarray, reference: tuple, dt: float, controller: LQRTrackingController) -> SimulationResult:
    """Run one LQR parameter setting from the same offset pose."""
    ref_x, ref_y, ref_heading, ref_v, ref_omega = reference
    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)
    distance_error = np.empty_like(time)
    commanded_v = np.empty_like(time)
    commanded_omega = np.empty_like(time)

    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y
        commanded_v[index], commanded_omega[index], distance_error[index], _ = controller.command(
            robot.state.x,
            robot.state.y,
            robot.state.heading,
            ref_x[index],
            ref_y[index],
            ref_heading[index],
            ref_v[index],
            ref_omega[index],
            dt,
        )
        robot.step(commanded_v[index], commanded_omega[index], dt)

    return SimulationResult(actual_x, actual_y, distance_error, commanded_v, commanded_omega)


def main() -> None:
    dt = 0.02
    time, ref_x, ref_y, ref_heading, ref_v, ref_omega = circle(
        radius=2.0, angular_speed=0.35, duration=8.0, dt=dt
    )
    reference = (ref_x, ref_y, ref_heading, ref_v, ref_omega)
    settings = {
        "baseline": LQRTrackingController(
            state_cost=(12.0, 12.0, 4.0), control_cost=(1.0, 0.5)
        ),
        "higher state cost": LQRTrackingController(
            state_cost=(40.0, 40.0, 10.0), control_cost=(1.0, 0.5)
        ),
        "higher control cost": LQRTrackingController(
            state_cost=(12.0, 12.0, 4.0), control_cost=(4.0, 2.0)
        ),
    }
    results = {name: simulate(time, reference, dt, controller) for name, controller in settings.items()}

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    path_axis, error_axis = axes[0]
    linear_velocity_axis, angular_velocity_axis = axes[1]
    colors = {"baseline": "tab:blue", "higher state cost": "tab:red", "higher control cost": "tab:green"}

    path_axis.plot(ref_x, ref_y, "--", color="black", label="reference circle")
    for name, result in results.items():
        path_axis.plot(result.x, result.y, color=colors[name], label=name)
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_title("LQR Trajectory Comparison")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend()

    for name, result in results.items():
        error_axis.plot(time, result.distance_error, color=colors[name], label=name)
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("distance error (m)")
    error_axis.set_title("Tracking Error")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend()

    for name, result in results.items():
        linear_velocity_axis.plot(time, result.commanded_v, color=colors[name], label=name)
    linear_velocity_axis.plot(time, ref_v, "--", color="black", label="reference")
    linear_velocity_axis.axhline(1.5, color="tab:red", linestyle=":", label="limit")
    linear_velocity_axis.set_xlabel("time (s)")
    linear_velocity_axis.set_ylabel("linear velocity (m/s)")
    linear_velocity_axis.set_title("Linear-Velocity Commands")
    linear_velocity_axis.grid(True, alpha=0.3)
    linear_velocity_axis.legend()

    for name, result in results.items():
        angular_velocity_axis.plot(time, result.commanded_omega, color=colors[name], label=name)
    angular_velocity_axis.plot(time, ref_omega, "--", color="black", label="reference")
    angular_velocity_axis.axhline(2.5, color="tab:red", linestyle=":", label="limits")
    angular_velocity_axis.axhline(-2.5, color="tab:red", linestyle=":")
    angular_velocity_axis.set_xlabel("time (s)")
    angular_velocity_axis.set_ylabel("angular velocity (rad/s)")
    angular_velocity_axis.set_title("Angular-Velocity Commands")
    angular_velocity_axis.grid(True, alpha=0.3)
    angular_velocity_axis.legend()

    figure.tight_layout()
    output_path = Path("outputs/lqr_tuning_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")
    for name, result in results.items():
        print(f"{name} final distance error: {result.distance_error[-1]:.3f} m")


if __name__ == "__main__":
    main()
