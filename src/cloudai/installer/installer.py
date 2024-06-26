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

import logging
from typing import Iterable

from cloudai._core.install_status_result import InstallStatusResult
from cloudai._core.registry import Registry
from cloudai._core.system import System
from cloudai._core.test import Test


class Installer:
    """
    A wrapper class that creates and manages a specific installer instance based on the system's configuration.

    This class facilitates the initialization of the appropriate installer based on the system.

    Attributes
        installer (BaseInstaller): The specific installer instance for the system.
    """

    def __init__(self, system: System):
        """
        Initialize the Installer with a system object and installation path.

        Args:
            system (System): The system schema object.
        """
        scheduler_type = system.scheduler
        registry = Registry()
        installer_class = registry.installers_map.get(scheduler_type)
        if installer_class is None:
            raise NotImplementedError(f"No installer available for scheduler: {scheduler_type}")
        self.installer = installer_class(system)

    def is_installed(self, tests: Iterable[Test]) -> InstallStatusResult:
        """
        Check if the necessary components for the provided test templates are already installed.

        Args:
            tests (Iterable[Test]): The list of test templates to check.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        logging.info("Checking installation status of components.")
        return self.installer.is_installed(tests)

    def install(self, tests: Iterable[Test]) -> InstallStatusResult:
        """
        Check if the necessary components are installed and install them if not using the instantiated installer.

        Args:
            tests (Iterable[Test]): The list of test templates to install.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        logging.info("Installing test templates.")
        return self.installer.install(tests)

    def uninstall(self, tests: Iterable[Test]) -> InstallStatusResult:
        """
        Uninstall the benchmarks or test templates using the instantiated installer.

        Args:
            tests (Iterable[Test]): The list of test templates to uninstall.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        logging.info("Uninstalling test templates.")
        return self.installer.uninstall(tests)
