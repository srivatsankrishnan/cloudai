# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from .command_gen_strategy import CommandGenStrategy
from .grading_strategy import GradingStrategy
from .install_strategy import InstallStrategy
from .job_id_retrieval_strategy import JobIdRetrievalStrategy
from .report_generation_strategy import ReportGenerationStrategy
from .strategy_registry import StrategyRegistry
from .test_template_strategy import TestTemplateStrategy

__all__ = [
    "StrategyRegistry",
    "TestTemplateStrategy",
    "InstallStrategy",
    "CommandGenStrategy",
    "JobIdRetrievalStrategy",
    "ReportGenerationStrategy",
    "GradingStrategy",
]