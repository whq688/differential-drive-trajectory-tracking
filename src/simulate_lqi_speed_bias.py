"""Show how LQI removes LQR's persistent error under a speed bias."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.controllers import LQITrackingController
from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


def main() -> None:
    dt = 0.02
    speed_scale = 1.15
    time, ref_x, ref_y, ref_heading, ref_v, ref_omega = circle(
        radius=2.0, angular_speed=0.35, duration=24.0, dt=dt
    )
    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    controller = LQITrackingController()

    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)
    distance_error = np.empty_like(time)
    along_track_error = np.empty_like(time)
    integrated_error = np.empty_like(time)
    commanded_v = np.empty_like(time)

    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y
        (
            commanded_v[index],
            command_omega,
            distance_error[index],
            along_track_error[index],
            _,
        ) = controller.command(
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
        integrated_error[index] = controller.integrated_along_track_error
        robot.step(speed_scale * commanded_v[index], command_omega, dt)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    path_axis, error_axis, integral_axis = axes
    path_axis.plot(ref_x, ref_y, "--", label="reference circle")
    path_axis.plot(
        actual_x,
        actual_y,
        label=f"LQI robot ({speed_scale:.0%} actual speed)",
    )
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_title("LQI Compensates a Persistent Speed Bias")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend()

    error_axis.plot(time, distance_error, label="distance error")
    error_axis.plot(time, along_track_error, label="along-track error")
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("error (m)")
    error_axis.set_title("Position Error")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend()

    integral_axis.plot(time, integrated_error)
    integral_axis.set_xlabel("time (s)")
    integral_axis.set_ylabel("integrated error (m-s)")
    integral_axis.set_title("The LQI Integral State")
    integral_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path = Path("outputs/lqi_speed_bias.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")
    print(f"Final distance error: {distance_error[-1]:.3f} m")
    print(f"Final along-track error: {along_track_error[-1]:.3f} m")
    print(f"Final commanded speed: {commanded_v[-1]:.3f} m/s")
    print(f"Final actual speed: {speed_scale * commanded_v[-1]:.3f} m/s")


if __name__ == "__main__":
    main()
