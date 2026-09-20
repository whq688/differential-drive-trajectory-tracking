"""Reference trajectories for simulation experiments."""

import numpy as np


def circle(radius: float, angular_speed: float, duration: float, dt: float):
    """Return circular-reference poses and matching feed-forward commands.

    The robot starts at (radius, 0), pointed upward, and travels counter-clockwise.
    """
    time = np.arange(0.0, duration + dt, dt)
    phase = angular_speed * time
    x = radius * np.cos(phase)
    y = radius * np.sin(phase)
    heading = phase + np.pi / 2
    linear_velocity = np.full_like(time, radius * angular_speed)
    angular_velocity = np.full_like(time, angular_speed)
    return time, x, y, heading, linear_velocity, angular_velocity
