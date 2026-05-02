# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import logging
import random as stdlib_random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cloudai.core import BaseRunner, Registry, TestRun
from cloudai.util.lazy_imports import lazy

from .backends.gym_server import GymServer, GymServerBackend, create_gym_server  # noqa: F401
from .backends.protocols import StepBackend, TrajectoryEntry
from .backends.runner import RunnerBackend, values_match_exact
from .base_gym import BaseGym


class CloudAIGymEnv(BaseGym):
    """
    Unified Gym environment for CloudAI.

    Supports three execution modes selected automatically:

    - **Runner mode** (default): launches real workloads via the CloudAI runner,
      reads metrics from job output.  Used for standard DSE.
    - **Online mode** (``live_rl_mode=true`` in cmd_args): delegates to an
      in-process GymServer for fast, stateful interaction.  Used for
      online RL / simulation-based optimization (e.g. kvpilot).
    - **Offline mode** (no test_run/runner): serves pre-loaded trajectory data
      for offline RL agents (e.g. CQL).  Use ``load_trajectory_files()`` to
      populate data after construction.

    Agents interact with the same interface regardless of mode.
    """

    def __init__(
        self,
        test_run: Optional[TestRun] = None,
        runner: Optional[BaseRunner] = None,
        gym_server: Optional[Any] = None,
    ):
        self.test_run = test_run
        self.runner = runner
        self.max_steps = test_run.test.agent_steps if test_run else 0
        self.reward_function = (
            Registry().get_reward_function(test_run.test.agent_reward_function) if test_run else lambda obs: 0.0
        )
        self._step_count = 0
        self._rng = stdlib_random.Random(42)
        self._trajectory: list[TrajectoryEntry] = []
        self._trajectory_by_iteration: dict[int, list[TrajectoryEntry]] = {}
        self._backend = self._resolve_backend(test_run, runner, gym_server)
        super().__init__()

    @staticmethod
    def _resolve_backend(
        test_run: Optional[TestRun],
        runner: Optional[BaseRunner],
        gym_server: Optional[Any],
    ) -> Optional[StepBackend]:
        """Select the execution backend based on the provided arguments."""
        if gym_server is not None:
            return GymServerBackend(gym_server)
        if test_run is None or runner is None:
            return None
        if getattr(test_run.test.cmd_args, "live_rl_mode", False):
            return GymServerBackend(create_gym_server(test_run))
        return RunnerBackend(test_run, runner)

    @property
    def _is_online(self) -> bool:
        return isinstance(self._backend, GymServerBackend)

    def define_action_space(self) -> Dict[str, Any]:
        if self._backend is None:
            return {}
        return self._backend.get_action_space()

    def define_observation_space(self) -> list:
        if self._backend is None:
            return [0.0]
        return self._backend.get_observation_space()

    @property
    def first_sweep(self) -> Any:
        space = self.define_action_space()
        if isinstance(space, dict) and space.get("type") == "continuous":
            shape = int(space.get("shape", 1))
            low = float(space.get("low", -1.0))
            return [low] * shape
        return {k: v[0] if isinstance(v, list) else v for k, v in space.items()}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,  # noqa: Vulture
    ) -> Tuple[list, dict[str, Any]]:
        if seed is not None:
            self._rng = stdlib_random.Random(seed)
        self._step_count = 0
        if self._backend is None:
            return [0.0], {}
        return self._backend.reset(seed)

    def step(self, action: Any) -> Tuple[list, float, bool, dict]:
        if self._backend is None:
            raise RuntimeError("step() is not available in offline mode")
        self._step_count += 1
        observation, done, info = self._backend.step(action)
        reward = self.reward_function(observation)

        entry = TrajectoryEntry(
            step=self._step_count,
            action=action,
            reward=reward,
            observation=observation,
            info=info,
        )
        self._write_trajectory(entry)

        if isinstance(self._backend, RunnerBackend):
            self._backend.cache_trajectory(entry)

        return observation, reward, done, info

    def render(self, mode: str = "human"):
        if self._is_online:
            logging.info(f"CloudAIGymEnv [online] step {self._step_count}")
        elif self.test_run is not None:
            print(f"Step {self.test_run.current_iteration}: Parameters {self.test_run.test.cmd_args}")

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = stdlib_random.Random(seed)
            lazy.np.random.seed(seed)

    def compute_reward(self, observation: list) -> float:
        return self.reward_function(observation)

    def get_observation(self, action: Any) -> list:
        if isinstance(self._backend, RunnerBackend):
            return self._backend.get_observation(action)
        if self._backend is not None:
            return self._backend.get_observation_space()
        return [0.0]

    _MAX_OBS_CSV_ELEMENTS = 1024

    def _write_trajectory(self, entry: TrajectoryEntry) -> None:
        self._trajectory.append(entry)
        self.current_trajectory.append(entry)

        file_exists = self.trajectory_file_path.exists()
        logging.debug(f"Writing trajectory into {self.trajectory_file_path} (exists: {file_exists})")
        self.trajectory_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.trajectory_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "action", "reward", "observation", "info"])
            obs = entry.observation
            if isinstance(obs, list) and len(obs) > self._MAX_OBS_CSV_ELEMENTS:
                obs = f"[truncated len={len(obs)}]"
            writer.writerow([entry.step, entry.action, entry.reward, obs, entry.info])

    def write_trajectory(self, entry: TrajectoryEntry) -> None:
        """Public method for external callers (e.g. single_sbatch_runner)."""
        self._write_trajectory(entry)

    @property
    def trajectory_file_path(self) -> Path:
        if self.test_run is None:
            return Path("trajectory.csv")
        if self._is_online:
            return self.test_run.output_path / "trajectory.csv"
        return self.runner.scenario_root / self.test_run.name / f"{self.test_run.current_iteration}" / "trajectory.csv"

    @property
    def current_trajectory(self) -> list[TrajectoryEntry]:
        iteration = self.test_run.current_iteration if self.test_run else 0
        return self._trajectory_by_iteration.setdefault(iteration, [])

    def get_cached_trajectory_result(self, action: Any) -> Optional[TrajectoryEntry]:
        """Return a cached entry matching the given action in the current iteration, or None."""
        for entry in self.current_trajectory:
            if values_match_exact(entry.action, action):
                return entry
        return None

    def get_all_trajectory_entries(self) -> list[TrajectoryEntry]:
        """
        Return all trajectory entries (pre-loaded + current session + explicitly loaded).

        Useful for offline RL agents that train on historical data.
        """
        entries = list(self._trajectory)
        if isinstance(self._backend, RunnerBackend):
            backend_entries = self._backend.get_all_entries()
            seen = {id(e) for e in entries}
            entries.extend(e for e in backend_entries if id(e) not in seen)
        return entries

    def load_trajectory_files(self, paths: list[Path]) -> int:
        """
        Load additional trajectory.csv files into the cache.

        Returns the number of entries loaded.
        """
        import ast

        loaded = 0
        for traj_file in paths:
            traj_path = Path(traj_file)
            if not traj_path.is_file():
                logging.warning("Trajectory file not found: %s", traj_path)
                continue
            with open(traj_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        entry = TrajectoryEntry(
                            step=int(row["step"]),
                            action=ast.literal_eval(row["action"]),
                            reward=float(row["reward"]),
                            observation=ast.literal_eval(row["observation"]),
                            info=ast.literal_eval(row.get("info", "{}")),
                        )
                        self._trajectory.append(entry)
                        loaded += 1
                    except (ValueError, SyntaxError) as exc:
                        logging.debug("Skipping malformed row in %s: %s", traj_path, exc)
        if loaded:
            logging.info("Loaded %d trajectory entries from %d file(s)", loaded, len(paths))
        return loaded
