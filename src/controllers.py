"""Simple feedback controllers used by the trajectory-tracking simulations."""

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve_discrete_are


@dataclass
class ProportionalTrackingController:
    """Track a reference point with proportional speed and heading feedback."""

    heading_gain: float = 2.5
    distance_gain: float = 0.8
    max_angular_velocity: float = 2.5
    max_linear_velocity: float = 1.5

    def command(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        reference_x: float,
        reference_y: float,
        reference_heading: float,
        reference_linear_velocity: float,
        reference_angular_velocity: float,
    ) -> tuple[float, float, float, float]:
        """Return linear velocity, angular velocity, distance and heading error."""
        error_x = reference_x - robot_x
        error_y = reference_y - robot_y
        distance_error = float(np.hypot(error_x, error_y))

        if distance_error < 1e-6:
            desired_heading = reference_heading
        else:
            desired_heading = float(np.arctan2(error_y, error_x))

        heading_error = self._wrap_angle(desired_heading - robot_heading)
        # A robot that is farther from the moving reference point is allowed
        # to travel faster, but only up to a physically plausible limit.
        linear_velocity = reference_linear_velocity + self.distance_gain * distance_error
        linear_velocity = float(np.clip(linear_velocity, 0.0, self.max_linear_velocity))

        angular_velocity = reference_angular_velocity + self.heading_gain * heading_error
        angular_velocity = float(
            np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        )
        return linear_velocity, angular_velocity, distance_error, heading_error

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Keep an angle in [-pi, pi)."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class PITrackingController:
    """Track a moving reference with proportional and integral speed feedback."""

    heading_gain: float = 2.5
    distance_gain: float = 0.8
    integral_gain: float = 0.10
    max_angular_velocity: float = 2.5
    max_linear_velocity: float = 1.5
    max_integral_error: float = 2.0
    integrated_along_track_error: float = field(default=0.0, init=False)

    def command(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        reference_x: float,
        reference_y: float,
        reference_heading: float,
        reference_linear_velocity: float,
        reference_angular_velocity: float,
        dt: float,
    ) -> tuple[float, float, float, float, float]:
        """Return commands plus distance, heading and along-track errors."""
        error_x = reference_x - robot_x
        error_y = reference_y - robot_y
        distance_error = float(np.hypot(error_x, error_y))

        if distance_error < 1e-6:
            desired_heading = reference_heading
        else:
            desired_heading = float(np.arctan2(error_y, error_x))
        heading_error = self._wrap_angle(desired_heading - robot_heading)

        # Project the position error onto the reference travel direction.
        # Positive means the robot is behind the moving reference point.
        along_track_error = float(
            error_x * np.cos(reference_heading) + error_y * np.sin(reference_heading)
        )
        self.integrated_along_track_error += along_track_error * dt
        self.integrated_along_track_error = float(
            np.clip(
                self.integrated_along_track_error,
                -self.max_integral_error,
                self.max_integral_error,
            )
        )

        linear_velocity = (
            reference_linear_velocity
            + self.distance_gain * along_track_error
            + self.integral_gain * self.integrated_along_track_error
        )
        linear_velocity = float(np.clip(linear_velocity, 0.0, self.max_linear_velocity))

        angular_velocity = reference_angular_velocity + self.heading_gain * heading_error
        angular_velocity = float(
            np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        )
        return (
            linear_velocity,
            angular_velocity,
            distance_error,
            heading_error,
            along_track_error,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Keep an angle in [-pi, pi)."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class PIDTrackingController:
    """Track a moving reference with PID speed feedback."""

    heading_gain: float = 2.5
    distance_gain: float = 0.8
    integral_gain: float = 0.10
    derivative_gain: float = 0.15
    derivative_filter_alpha: float = 0.15
    max_angular_velocity: float = 2.5
    max_linear_velocity: float = 1.5
    max_integral_error: float = 2.0
    integrated_along_track_error: float = field(default=0.0, init=False)
    previous_along_track_error: float | None = field(default=None, init=False)
    filtered_derivative_error: float = field(default=0.0, init=False)

    def command(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        reference_x: float,
        reference_y: float,
        reference_heading: float,
        reference_linear_velocity: float,
        reference_angular_velocity: float,
        dt: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return commands plus distance, heading, along-track and derivative errors."""
        error_x = reference_x - robot_x
        error_y = reference_y - robot_y
        distance_error = float(np.hypot(error_x, error_y))

        if distance_error < 1e-6:
            desired_heading = reference_heading
        else:
            desired_heading = float(np.arctan2(error_y, error_x))
        heading_error = self._wrap_angle(desired_heading - robot_heading)

        along_track_error = float(
            error_x * np.cos(reference_heading) + error_y * np.sin(reference_heading)
        )
        self.integrated_along_track_error += along_track_error * dt
        self.integrated_along_track_error = float(
            np.clip(
                self.integrated_along_track_error,
                -self.max_integral_error,
                self.max_integral_error,
            )
        )

        if self.previous_along_track_error is None:
            raw_derivative_error = 0.0
        else:
            raw_derivative_error = (
                along_track_error - self.previous_along_track_error
            ) / dt
        self.filtered_derivative_error = (
            self.derivative_filter_alpha * raw_derivative_error
            + (1.0 - self.derivative_filter_alpha) * self.filtered_derivative_error
        )
        self.previous_along_track_error = along_track_error

        linear_velocity = (
            reference_linear_velocity
            + self.distance_gain * along_track_error
            + self.integral_gain * self.integrated_along_track_error
            + self.derivative_gain * self.filtered_derivative_error
        )
        linear_velocity = float(np.clip(linear_velocity, 0.0, self.max_linear_velocity))

        angular_velocity = reference_angular_velocity + self.heading_gain * heading_error
        angular_velocity = float(
            np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        )
        return (
            linear_velocity,
            angular_velocity,
            distance_error,
            heading_error,
            along_track_error,
            raw_derivative_error,
            self.filtered_derivative_error,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Keep an angle in [-pi, pi)."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class LQRTrackingController:
    """Time-varying LQR tracker for the unicycle robot near a reference path."""

    state_cost: tuple[float, float, float] = (40.0, 40.0, 10.0)
    control_cost: tuple[float, float] = (1.0, 0.5)
    max_linear_velocity: float = 1.5
    max_angular_velocity: float = 2.5

    def command(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        reference_x: float,
        reference_y: float,
        reference_heading: float,
        reference_linear_velocity: float,
        reference_angular_velocity: float,
        dt: float,
    ) -> tuple[float, float, float, np.ndarray]:
        """Return LQR commands, distance error, and the feedback gain matrix."""
        # LQR uses actual-minus-reference error, so a negative feedback law
        # naturally moves the robot back toward the reference pose.
        state_error = np.array(
            [
                robot_x - reference_x,
                robot_y - reference_y,
                self._wrap_angle(robot_heading - reference_heading),
            ]
        )

        sin_heading = np.sin(reference_heading)
        cos_heading = np.cos(reference_heading)
        continuous_a = np.array(
            [
                [0.0, 0.0, -reference_linear_velocity * sin_heading],
                [0.0, 0.0, reference_linear_velocity * cos_heading],
                [0.0, 0.0, 0.0],
            ]
        )
        continuous_b = np.array(
            [
                [cos_heading, 0.0],
                [sin_heading, 0.0],
                [0.0, 1.0],
            ]
        )
        discrete_a = np.eye(3) + continuous_a * dt
        discrete_b = continuous_b * dt

        q = np.diag(self.state_cost)
        r = np.diag(self.control_cost)
        riccati_solution = solve_discrete_are(discrete_a, discrete_b, q, r)
        gain = np.linalg.solve(
            discrete_b.T @ riccati_solution @ discrete_b + r,
            discrete_b.T @ riccati_solution @ discrete_a,
        )

        control_correction = -gain @ state_error
        linear_velocity = float(
            np.clip(
                reference_linear_velocity + control_correction[0],
                0.0,
                self.max_linear_velocity,
            )
        )
        angular_velocity = float(
            np.clip(
                reference_angular_velocity + control_correction[1],
                -self.max_angular_velocity,
                self.max_angular_velocity,
            )
        )
        distance_error = float(np.hypot(state_error[0], state_error[1]))
        return linear_velocity, angular_velocity, distance_error, gain

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Keep an angle in [-pi, pi)."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class LQITrackingController:
    """Time-varying LQR tracker augmented with an along-track integral state."""

    state_cost: tuple[float, float, float] = (40.0, 40.0, 10.0)
    integral_state_cost: float = 2.0
    control_cost: tuple[float, float] = (1.0, 0.5)
    max_linear_velocity: float = 1.5
    max_angular_velocity: float = 2.5
    max_integrated_along_track_error: float = 3.0
    integrated_along_track_error: float = field(default=0.0, init=False)

    def command(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        reference_x: float,
        reference_y: float,
        reference_heading: float,
        reference_linear_velocity: float,
        reference_angular_velocity: float,
        dt: float,
    ) -> tuple[float, float, float, float, np.ndarray]:
        """Return LQI commands, errors, and the augmented feedback gain."""
        # This controller uses actual-minus-reference errors.  Along-track
        # error is therefore positive when the robot is ahead of the reference.
        state_error = np.array(
            [
                robot_x - reference_x,
                robot_y - reference_y,
                self._wrap_angle(robot_heading - reference_heading),
            ]
        )
        sin_heading = np.sin(reference_heading)
        cos_heading = np.cos(reference_heading)
        along_track_error = float(
            state_error[0] * cos_heading + state_error[1] * sin_heading
        )
        self.integrated_along_track_error += along_track_error * dt
        self.integrated_along_track_error = float(
            np.clip(
                self.integrated_along_track_error,
                -self.max_integrated_along_track_error,
                self.max_integrated_along_track_error,
            )
        )

        continuous_a = np.array(
            [
                [0.0, 0.0, -reference_linear_velocity * sin_heading],
                [0.0, 0.0, reference_linear_velocity * cos_heading],
                [0.0, 0.0, 0.0],
            ]
        )
        continuous_b = np.array(
            [
                [cos_heading, 0.0],
                [sin_heading, 0.0],
                [0.0, 1.0],
            ]
        )
        discrete_a = np.eye(3) + continuous_a * dt
        discrete_b = continuous_b * dt

        # z(k+1) = z(k) + dt * [cos(theta_ref), sin(theta_ref), 0] * e(k)
        along_track_projection = np.array([cos_heading, sin_heading, 0.0])
        augmented_a = np.zeros((4, 4))
        augmented_a[:3, :3] = discrete_a
        augmented_a[3, :3] = dt * along_track_projection
        augmented_a[3, 3] = 1.0
        augmented_b = np.zeros((4, 2))
        augmented_b[:3, :] = discrete_b

        q = np.diag((*self.state_cost, self.integral_state_cost))
        r = np.diag(self.control_cost)
        riccati_solution = solve_discrete_are(augmented_a, augmented_b, q, r)
        gain = np.linalg.solve(
            augmented_b.T @ riccati_solution @ augmented_b + r,
            augmented_b.T @ riccati_solution @ augmented_a,
        )
        augmented_error = np.append(state_error, self.integrated_along_track_error)
        control_correction = -gain @ augmented_error

        linear_velocity = float(
            np.clip(
                reference_linear_velocity + control_correction[0],
                0.0,
                self.max_linear_velocity,
            )
        )
        angular_velocity = float(
            np.clip(
                reference_angular_velocity + control_correction[1],
                -self.max_angular_velocity,
                self.max_angular_velocity,
            )
        )
        distance_error = float(np.hypot(state_error[0], state_error[1]))
        return (
            linear_velocity,
            angular_velocity,
            distance_error,
            along_track_error,
            gain,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Keep an angle in [-pi, pi)."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
