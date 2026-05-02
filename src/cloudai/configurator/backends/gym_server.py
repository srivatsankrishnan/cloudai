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

import importlib
from typing import Any, Dict, List, Optional, Protocol, Tuple

from cloudai.core import TestRun


class GymServer(Protocol):
    """Protocol for gym server objects used in online mode."""

    def reset(self) -> Tuple[List[float], Dict[str, Any]]: ...
    def step(self, action: Dict[str, Any]) -> Tuple[List[float], float, bool, Dict[str, Any]]: ...
    def get_action_space(self) -> Dict[str, Any]: ...
    def get_observation_space(self) -> List[float]: ...


class GymServerBackend:
    """Backend that delegates to an in-process GymServer for fast, stateful interaction."""

    def __init__(self, server: Any) -> None:
        self._server = server
        self._step_count = 0

    def get_action_space(self) -> Dict[str, Any]:
        return self._server.get_action_space()

    def get_observation_space(self) -> list:
        return self._server.get_observation_space()

    def reset(self, seed: Optional[int] = None) -> Tuple[list, dict[str, Any]]:
        self._step_count = 0
        return self._server.reset()

    def step(self, action: Any) -> Tuple[list, bool, dict[str, Any]]:
        self._step_count += 1
        observation, _raw_reward, done, info = self._server.step(action)
        return observation, done, info


def create_gym_server(test_run: TestRun) -> Any:
    """Instantiate a GymServer from the env_class path in cmd_args."""
    import inspect

    from cloudai.util import flatten_dict

    cmd_args = test_run.test.cmd_args
    args_dict = flatten_dict(cmd_args.model_dump())

    env_class_path = args_dict.pop("env_class", None)
    if not env_class_path:
        raise ValueError("online mode requires 'env_class' in cmd_args pointing to a GymServer class")

    for key in ("live_rl_mode", "docker_image_url"):
        args_dict.pop(key, None)

    module_path, class_name = env_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    server_cls = getattr(module, class_name)

    sig = inspect.signature(server_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    if valid_params and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        args_dict = {k: v for k, v in args_dict.items() if k in valid_params}

    return server_cls(**args_dict)
