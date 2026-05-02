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

import copy
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cloudai.core import METRIC_ERROR, BaseRunner, TestRun
from cloudai.util.lazy_imports import lazy

from .protocols import TrajectoryEntry


def values_match_exact(left: Any, right: Any) -> bool:
    """Deep-compare two values (dicts, lists, scalars) for exact equality."""
    if type(left) is not type(right):
        return False
    elif isinstance(left, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(values_match_exact(left[key], right[key]) for key in left)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(values_match_exact(lv, rv) for lv, rv in zip(left, right, strict=True))
    else:
        return left == right


class RunnerBackend:
    """Backend that launches real workloads via the CloudAI runner."""

    def __init__(self, test_run: TestRun, runner: BaseRunner) -> None:
        self._test_run = test_run
        self._original_test_run = copy.deepcopy(test_run)
        self._runner = runner
        self._trajectory_cache: dict[int, list[TrajectoryEntry]] = {}
        self._load_trajectory_cache()

    @property
    def test_run(self) -> TestRun:
        return self._test_run

    @test_run.setter
    def test_run(self, value: TestRun) -> None:
        self._test_run = value

    def get_action_space(self) -> Dict[str, Any]:
        return self._test_run.param_space

    def get_observation_space(self) -> list:
        n_metrics = max(len(self._test_run.test.agent_metrics), 1)
        return [0.0] * n_metrics

    def reset(self, seed: Optional[int] = None) -> Tuple[list, dict[str, Any]]:
        if seed is not None:
            lazy.np.random.seed(seed)
        self._test_run.current_iteration = 0
        return self.get_observation_space(), {}

    def step(self, action: Any) -> Tuple[list, bool, dict[str, Any]]:
        self._test_run = self._test_run.apply_params_set(action)

        cached = self._get_cached_result(action)
        if cached is not None:
            logging.info("Retrieved cached result with reward %s. Skipping step.", cached.reward)
            return cached.observation, False, cached.info

        if not self._test_run.test.constraint_check(self._test_run, self._runner.system):
            logging.info("Constraint check failed. Skipping step.")
            return [-1.0], True, {"reason": "constraint_check_failed"}

        new_tr = copy.deepcopy(self._test_run)
        new_tr.output_path = self._runner.get_job_output_path(new_tr)
        self._runner.test_scenario.test_runs = [new_tr]

        self._runner.shutting_down = False
        self._runner.jobs.clear()
        self._runner.testrun_to_job_map.clear()

        try:
            self._runner.run()
        except Exception as e:
            logging.error(f"Error running step {self._test_run.step}: {e}")

        if self._runner.test_scenario.test_runs and self._runner.test_scenario.test_runs[0].output_path.exists():
            self._test_run = self._runner.test_scenario.test_runs[0]
        else:
            self._test_run = copy.deepcopy(self._original_test_run)
            self._test_run.step = new_tr.step
            self._test_run.output_path = new_tr.output_path

        observation = self._get_observation(action)
        return observation, False, {}

    def get_observation(self, action: Any) -> list:
        return self._get_observation(action)

    def _get_observation(self, action: Any) -> list:
        all_metrics = self._test_run.test.agent_metrics
        if not all_metrics:
            raise ValueError("No agent metrics defined for the test run")

        observation = []
        for metric in all_metrics:
            v = self._test_run.get_metric_value(self._runner.system, metric)
            if v == METRIC_ERROR:
                v = -1.0
            observation.append(v)
        return observation

    def cache_trajectory(self, entry: TrajectoryEntry) -> None:
        self._trajectory_cache.setdefault(self._test_run.current_iteration, []).append(entry)

    def _get_cached_result(self, action: Any) -> Optional[TrajectoryEntry]:
        for entry in self._trajectory_cache.get(self._test_run.current_iteration, []):
            if values_match_exact(entry.action, action):
                return entry
        return None

    def get_all_entries(self) -> list[TrajectoryEntry]:
        """Return all cached trajectory entries across iterations."""
        return [e for entries in self._trajectory_cache.values() for e in entries]

    def _load_trajectory_cache(self) -> None:
        """Pre-load trajectory.csv files from previous runs into the cache."""
        import ast

        scenario_root = getattr(self._runner, "scenario_root", None)
        if not isinstance(scenario_root, Path):
            return

        tr_name = self._test_run.name
        tr_dir = scenario_root / tr_name
        if not tr_dir.is_dir():
            return

        loaded = 0
        for iteration_dir in sorted(tr_dir.iterdir()):
            if not iteration_dir.is_dir():
                continue
            traj_file = iteration_dir / "trajectory.csv"
            if not traj_file.is_file():
                continue
            try:
                iteration = int(iteration_dir.name)
            except ValueError:
                continue

            with open(traj_file, newline="") as f:
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
                        self._trajectory_cache.setdefault(iteration, []).append(entry)
                        loaded += 1
                    except (ValueError, SyntaxError) as exc:
                        logging.debug("Skipping malformed trajectory row in %s: %s", traj_file, exc)

        if loaded:
            logging.info("Pre-loaded %d trajectory entries from %s", loaded, tr_dir)
