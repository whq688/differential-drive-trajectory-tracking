"""Track a circular path with a time-varying LQR controller."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.controllers import LQRTrackingController
from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


def main() -> None:
    dt = 0.02
    time, ref_x, ref_y, ref_heading, ref_v, ref_omega = circle(
        radius=2.0,
        angular_speed=0.35,
        duration=24.0,
        dt=dt,
    )
    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    controller = LQRTrackingController()

    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)
    distance_error = np.empty_like(time)
    commanded_v = np.empty_like(time)
    commanded_omega = np.empty_like(time)
    gain_history = np.empty((len(time), 2, 3))

    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y
        commanded_v[index], commanded_omega[index], distance_error[index], gain_history[index] = controller.command(
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

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    path_axis, error_axis = axes[0]
    linear_velocity_axis, angular_velocity_axis = axes[1]
    path_axis.plot(ref_x, ref_y, "--", label="reference circle")
    path_axis.plot(actual_x, actual_y, label="LQR-controlled robot")
    path_axis.scatter(ref_x[0], ref_y[0], color="green", label="reference start", zorder=3)
    path_axis.scatter(actual_x[0], actual_y[0], color="red", label="robot start", zorder=3)
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_title("Time-Varying LQR Trajectory Tracking")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend()

    error_axis.plot(time, distance_error)
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("distance error (m)")
    error_axis.set_title("Distance to the Moving Reference")
    error_axis.grid(True, alpha=0.3)

    linear_velocity_axis.plot(time, ref_v, "--", label="reference linear velocity")
    linear_velocity_axis.plot(time, commanded_v, label="LQR command")
    linear_velocity_axis.axhline(
        controller.max_linear_velocity,
        color="tab:red",
        linestyle=":",
        label="linear-velocity limit",
    )
    linear_velocity_axis.set_xlabel("time (s)")
    linear_velocity_axis.set_ylabel("linear velocity (m/s)")
    linear_velocity_axis.set_title("Linear-Velocity Command")
    linear_velocity_axis.grid(True, alpha=0.3)
    linear_velocity_axis.legend()

    angular_velocity_axis.plot(time, ref_omega, "--", label="reference angular velocity")
    angular_velocity_axis.plot(time, commanded_omega, label="LQR command")
    angular_velocity_axis.axhline(
        controller.max_angular_velocity,
        color="tab:red",
        linestyle=":",
        label="angular-velocity limits",
    )
    angular_velocity_axis.axhline(
        -controller.max_angular_velocity,
        color="tab:red",
        linestyle=":",
    )
    angular_velocity_axis.set_xlabel("time (s)")
    angular_velocity_axis.set_ylabel("angular velocity (rad/s)")
    angular_velocity_axis.set_title("Angular-Velocity Command")
    angular_velocity_axis.grid(True, alpha=0.3)
    angular_velocity_axis.legend()

    figure.tight_layout()
    output_path = Path("outputs/lqr_tracking.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")
    print(f"Final distance error: {distance_error[-1]:.3f} m")
    print("Initial LQR gain K:")
    print(gain_history[0])


if __name__ == "__main__":
    main()
