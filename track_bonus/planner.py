"""Race-assist high-level planner for the 200 m track bonus.

Keeps the official interface:

    5D track observation -> [vx, vy, yaw_rate]

Compared with the previous robust planner, this version adds a small race mode:
it gives extra speed only when the robot is centered and aligned, and it boosts
turning/lateral correction when the robot is drifting away from the centerline.
The goal is to shave time without sacrificing the full-lap reliability that the
169 s config already has.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"

    speed_mps: float = 1.60
    turn_speed_mps: float = 1.08
    min_speed_mps: float = 0.38

    max_lateral_speed_mps: float = 0.30
    max_yaw_rate_radps: float = 1.85

    k_heading: float = 1.48
    k_lateral: float = 0.50
    k_lateral_speed: float = 0.34

    heading_slowdown: float = 0.14
    lateral_slowdown: float = 0.10
    curvature_slowdown: float = 0.00

    command_smoothing: float = 0.03
    stand_seconds: float = 0.15

    # Extra race-assist fields.  Existing configs can omit these.
    centered_speed_boost_mps: float = 0.08
    centered_lateral_norm: float = 0.18
    centered_heading_rad: float = 0.18
    lateral_rescue_boost: float = 0.55
    heading_rescue_boost: float = 0.35
    drift_slowdown: float = 0.18
    max_speed_mps: float = 1.75

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "turn_speed_mps": self.turn_speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "k_lateral_speed": self.k_lateral_speed,
            "heading_slowdown": self.heading_slowdown,
            "lateral_slowdown": self.lateral_slowdown,
            "curvature_slowdown": self.curvature_slowdown,
            "command_smoothing": self.command_smoothing,
            "stand_seconds": self.stand_seconds,
            "centered_speed_boost_mps": self.centered_speed_boost_mps,
            "centered_lateral_norm": self.centered_lateral_norm,
            "centered_heading_rad": self.centered_heading_rad,
            "lateral_rescue_boost": self.lateral_rescue_boost,
            "heading_rescue_boost": self.heading_rescue_boost,
            "drift_slowdown": self.drift_slowdown,
            "max_speed_mps": self.max_speed_mps,
        }


class StarterTrackPlanner:
    """Curvature-aware Stanley follower with dynamic race assistance."""

    def __init__(self, config: StarterPlannerConfig) -> None:
        if config.planner_type != "starter_pd":
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self._last_cmd = np.zeros(3, dtype=np.float32)
        self._last_lateral_error = 0.0

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        return cls(StarterPlannerConfig.load(path))

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            self._last_cmd = np.zeros(3, dtype=np.float32)
            self._last_lateral_error = 0.0
            return self._last_cmd.copy()

        raw_cmd, rescue_level = self._command_and_rescue_level(obs)

        # Keep commands smooth on easy sections, but react faster when the robot
        # starts drifting or pointing away from the tangent.
        base_alpha = float(np.clip(self.config.command_smoothing, 0.0, 0.95))
        alpha = base_alpha * (1.0 - 0.65 * rescue_level)
        cmd = (1.0 - alpha) * raw_cmd + alpha * self._last_cmd
        self._last_cmd = cmd.astype(np.float32)
        return self._last_cmd.copy()

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        cmd, _ = self._command_and_rescue_level(obs)
        return cmd

    def _command_and_rescue_level(
        self, obs: TrackControllerObservation
    ) -> tuple[np.ndarray, float]:
        cfg = self.config
        half_width = max(float(self.track.half_width_m), 1e-6)

        lateral_error = float(obs.lateral_error_norm) * half_width
        lateral_frac = min(abs(lateral_error) / half_width, 1.0)
        lateral_change = abs(lateral_error) - abs(self._last_lateral_error)
        self._last_lateral_error = lateral_error
        drifting_out = max(lateral_change / half_width, 0.0)

        curvature_norm = float(obs.curvature_norm)
        curvature_frac = min(abs(curvature_norm), 1.0)
        curvature = curvature_norm / max(float(self.track.turn_radius_m), 1e-6)

        heading_abs = min(abs(float(obs.heading_error_rad)), math.pi)
        centered_lateral = max(float(cfg.centered_lateral_norm), 1e-6)
        centered_heading = max(float(cfg.centered_heading_rad), 1e-6)
        centered_score = max(
            0.0,
            1.0
            - lateral_frac / centered_lateral
            - heading_abs / centered_heading,
        )

        scheduled_speed = (
            (1.0 - curvature_frac) * float(cfg.speed_mps)
            + curvature_frac * float(cfg.turn_speed_mps)
        )
        scheduled_speed += float(cfg.centered_speed_boost_mps) * centered_score

        heading_scale = 1.0 - float(cfg.heading_slowdown) * heading_abs / math.pi
        lateral_scale = 1.0 - float(cfg.lateral_slowdown) * lateral_frac
        curvature_scale = 1.0 - float(cfg.curvature_slowdown) * curvature_frac
        drift_scale = 1.0 - float(cfg.drift_slowdown) * min(drifting_out * 8.0, 1.0)
        vx = scheduled_speed * heading_scale * lateral_scale * curvature_scale * drift_scale
        vx = float(np.clip(vx, float(cfg.min_speed_mps), float(cfg.max_speed_mps)))

        rescue_level = float(
            np.clip(0.75 * lateral_frac + 0.50 * heading_abs + 3.0 * drifting_out, 0.0, 1.0)
        )
        k_lateral = float(cfg.k_lateral) * (1.0 + float(cfg.lateral_rescue_boost) * rescue_level)
        k_heading = float(cfg.k_heading) * (1.0 + float(cfg.heading_rescue_boost) * rescue_level)
        k_lateral_speed = float(cfg.k_lateral_speed) * (1.0 + 0.35 * rescue_level)

        lateral_bias = math.atan2(k_lateral * lateral_error, max(vx, 1e-3))
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        vy = np.clip(
            -k_lateral_speed * lateral_error,
            -float(cfg.max_lateral_speed_mps),
            float(cfg.max_lateral_speed_mps),
        )

        yaw_ff = curvature * vx
        yaw_fb = k_heading * heading_error
        yaw_rate = np.clip(
            yaw_ff + yaw_fb,
            -float(cfg.max_yaw_rate_radps),
            float(cfg.max_yaw_rate_radps),
        )

        return np.asarray([vx, vy, yaw_rate], dtype=np.float32), rescue_level
