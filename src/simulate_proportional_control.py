"""Track a circular trajectory with proportional speed and heading feedback."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.controllers import ProportionalTrackingController
from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


def main() -> None:
    dt = 0.02
    time, ref_x, ref_y, ref_heading, ref_linear_velocity, ref_angular_velocity = circle(
        radius=2.0,
        angular_speed=0.35,
        duration=18.0,
        dt=dt,
    )

    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    controller = ProportionalTrackingController(heading_gain=2.5, distance_gain=0.2)
    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)
    distance_error = np.empty_like(time)

    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y

        linear_velocity, angular_velocity, distance_error[index], _ = controller.command(
            robot_x=robot.state.x,
            robot_y=robot.state.y,
            robot_heading=robot.state.heading,
            reference_x=ref_x[index],
            reference_y=ref_y[index],
            reference_heading=ref_heading[index],
            reference_linear_velocity=ref_linear_velocity[index],
            reference_angular_velocity=ref_angular_velocity[index],
        )
        robot.step(linear_velocity, angular_velocity, dt)

    figure, (path_axis, error_axis) = plt.subplots(1, 2, figsize=(12, 5))
    path_axis.plot(ref_x, ref_y, "--", label="reference circle")
    path_axis.plot(actual_x, actual_y, label="P-controlled robot")
    path_axis.scatter(ref_x[0], ref_y[0], color="green", label="reference start", zorder=3)
    path_axis.scatter(actual_x[0], actual_y[0], color="red", label="robot start", zorder=3)
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_title("Trajectory Tracking with P Control")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend()

    error_axis.plot(time, distance_error)
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("distance error (m)")
    error_axis.set_title("Distance to the Moving Reference")
    error_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path = Path("outputs/proportional_control.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")
    print(f"Initial distance error: {distance_error[0]:.3f} m")
    print(f"Final distance error: {distance_error[-1]:.3f} m")


if __name__ == "__main__":
    main()
