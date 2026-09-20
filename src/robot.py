"""Kinematic model for a differential-drive robot."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RobotState:
    """Robot pose in the 2D world; heading is in radians."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0


class DifferentialDriveRobot:
    """Integrates the unicycle form of a differential-drive robot model."""

    def __init__(self, state: RobotState | None = None) -> None:
        self.state = state or RobotState()

    def step(self, linear_velocity: float, angular_velocity: float, dt: float) -> RobotState:
        """Advance the pose by one Euler-integration time step."""
        self.state.x += linear_velocity * np.cos(self.state.heading) * dt
        self.state.y += linear_velocity * np.sin(self.state.heading) * dt
        self.state.heading = self._wrap_angle(self.state.heading + angular_velocity * dt)
        return self.state

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Keep an angle in [-pi, pi)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
