"""Use PID control to track a circular trajectory despite a speed bias."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.controllers import PIDTrackingController
from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


def main() -> None:
    dt = 0.02
    speed_scale = 0.85
    time, ref_x, ref_y, ref_heading, ref_linear_velocity, ref_angular_velocity = circle(
        radius=2.0,
        angular_speed=0.35,
        duration=24.0,
        dt=dt,
    )

    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    controller = PIDTrackingController(
        heading_gain=2.5,
        distance_gain=0.8,
        integral_gain=0.10,
        derivative_gain=0.15,
    )
    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)
    distance_error = np.empty_like(time)
    raw_derivative_error = np.empty_like(time)
    filtered_derivative_error = np.empty_like(time)

    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y
        (
            commanded_v,
            angular_velocity,
            distance_error[index],
            _,
            _,
            raw_derivative_error[index],
            filtered_derivative_error[index],
        ) = controller.command(
            robot_x=robot.state.x,
            robot_y=robot.state.y,
            robot_heading=robot.state.heading,
            reference_x=ref_x[index],
            reference_y=ref_y[index],
            reference_heading=ref_heading[index],
            reference_linear_velocity=ref_linear_velocity[index],
            reference_angular_velocity=ref_angular_velocity[index],
            dt=dt,
        )
        robot.step(speed_scale * commanded_v, angular_velocity, dt)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    path_axis, error_axis, derivative_axis = axes
    path_axis.plot(ref_x, ref_y, "--", label="reference circle")
    path_axis.plot(actual_x, actual_y, label="PID-controlled robot (85% speed)")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_title("PID Control with Speed Bias")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend()

    error_axis.plot(time, distance_error)
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("distance error (m)")
    error_axis.set_title("Distance Error")
    error_axis.grid(True, alpha=0.3)

    derivative_axis.plot(time, raw_derivative_error, alpha=0.45, label="raw derivative")
    derivative_axis.plot(time, filtered_derivative_error, label="filtered derivative")
    derivative_axis.set_xlabel("time (s)")
    derivative_axis.set_ylabel("error rate (m/s)")
    derivative_axis.set_title("D Term: Filtered Error Change Rate")
    derivative_axis.grid(True, alpha=0.3)
    derivative_axis.legend()

    figure.tight_layout()
    output_path = Path("outputs/pid_control_with_speed_bias.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")
    print(f"Final distance error: {distance_error[-1]:.3f} m")


if __name__ == "__main__":
    main()
