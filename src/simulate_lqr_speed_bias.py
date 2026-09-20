"""Show the steady tracking error of ordinary LQR under a persistent speed bias."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.controllers import LQRTrackingController
from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


def main() -> None:
    dt = 0.02
    speed_scale = 0.85
    time, ref_x, ref_y, ref_heading, ref_v, ref_omega = circle(
        radius=2.0, angular_speed=0.35, duration=24.0, dt=dt
    )
    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    controller = LQRTrackingController()

    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)
    distance_error = np.empty_like(time)
    commanded_v = np.empty_like(time)

    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y
        commanded_v[index], command_omega, distance_error[index], _ = controller.command(
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
        robot.step(speed_scale * commanded_v[index], command_omega, dt)

    figure, (path_axis, error_axis) = plt.subplots(1, 2, figsize=(13, 5))
    path_axis.plot(ref_x, ref_y, "--", label="reference circle")
    path_axis.plot(actual_x, actual_y, label="LQR robot (85% actual speed)")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_title("LQR with a Persistent Speed Bias")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend()

    error_axis.plot(time, distance_error)
    error_axis.set_xlabel("time (s)")
    error_axis.set_ylabel("distance error (m)")
    error_axis.set_title("LQR Leaves a Persistent Error")
    error_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path = Path("outputs/lqr_speed_bias.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")
    print(f"Final distance error: {distance_error[-1]:.3f} m")
    print(f"Final commanded speed: {commanded_v[-1]:.3f} m/s")
    print(f"Final actual speed: {speed_scale * commanded_v[-1]:.3f} m/s")


if __name__ == "__main__":
    main()
