"""Run the Day 1 open-loop differential-drive simulation."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.robot import DifferentialDriveRobot, RobotState
from src.trajectories import circle


def main() -> None:
    dt = 0.02
    time, ref_x, ref_y, _, linear_velocity, angular_velocity = circle(
        radius=2.0,
        angular_speed=0.35,
        duration=18.0,
        dt=dt,
    )

    robot = DifferentialDriveRobot(RobotState(x=2.5, y=-0.4, heading=np.pi / 2))
    actual_x = np.empty_like(time)
    actual_y = np.empty_like(time)

    for index in range(len(time)):
        actual_x[index] = robot.state.x
        actual_y[index] = robot.state.y
        robot.step(linear_velocity[index], angular_velocity[index], dt)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot(ref_x, ref_y, "--", label="reference circle")
    axis.plot(actual_x, actual_y, label="open-loop robot")
    axis.scatter(ref_x[0], ref_y[0], color="green", label="start", zorder=3)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Differential-Drive Robot: Open-Loop Circle")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path = Path("outputs/open_loop_trajectory.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
