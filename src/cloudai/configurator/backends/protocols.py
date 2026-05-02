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

import dataclasses
from typing import Any, Dict, Optional, Protocol, Tuple


@dataclasses.dataclass(frozen=True)
class TrajectoryEntry:
    """Represents a trajectory entry."""

    step: int
    action: dict[str, Any]
    reward: float
    observation: list
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class StepBackend(Protocol):
    """Internal protocol for execution backends."""

    def get_action_space(self) -> Dict[str, Any]: ...
    def get_observation_space(self) -> list: ...
    def reset(self, seed: Optional[int] = None) -> Tuple[list, dict[str, Any]]: ...
    def step(self, action: Any) -> Tuple[list, bool, dict[str, Any]]: ...
