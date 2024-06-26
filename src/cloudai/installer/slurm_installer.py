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

import contextlib
import os
import shutil
import subprocess
from typing import Iterable, cast

import toml

from cloudai._core.base_installer import BaseInstaller
from cloudai._core.install_status_result import InstallStatusResult
from cloudai._core.system import System
from cloudai._core.test import Test
from cloudai.systems import SlurmSystem


class SlurmInstaller(BaseInstaller):
    """
    Installer for systems that use the Slurm scheduler.

    Handles the installation of benchmarks or test templates for Slurm-managed systems.

    Attributes
        CONFIG_FILE_NAME (str): The name of the configuration file.
        PREREQUISITES (List[str]): A list of required binaries for the installer.
        REQUIRED_SRUN_OPTIONS (List[str]): A list of required srun options to check.
        install_path (str): Path where the benchmarks are to be installed. This is optional since uninstallation does
            not require it.
        config_path (str): Path to the installation configuration file.
    """

    CONFIG_FILE_NAME = ".cloudai.toml"
    PREREQUISITES = ["git", "sbatch", "sinfo", "squeue", "srun", "scancel"]
    REQUIRED_SRUN_OPTIONS = [
        "--mpi",
        "--gpus-per-node",
        "--ntasks-per-node",
        "--container-image",
        "--container-mounts",
    ]

    def __init__(self, system: System):
        """
        Initialize the BaseInstaller with a system object and an optional installation path.

        Args:
            system (System): The system schema object.
        """
        super().__init__(system)
        slurm_system = cast(SlurmSystem, self.system)
        self.install_path = slurm_system.install_path
        self.config_path = os.path.join(os.path.expanduser("~"), self.CONFIG_FILE_NAME)

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check for the presence of required binaries and specific srun options, raising an error if any are missing.

        This ensures the system environment is properly set up before proceeding with the installation or uninstallation
        processes.

        Returns
            InstallStatusResult: Result containing the status and any error message.
        """
        try:
            super()._check_prerequisites()  # TODO: if fails, print out missing prerequisites
            self._check_required_binaries()
            self._check_srun_options()
            return InstallStatusResult(True)
        except EnvironmentError as e:
            return InstallStatusResult(False, str(e))

    def _check_required_binaries(self) -> None:
        """Check for the presence of required binaries, raising an error if any are missing."""
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")

    def _check_srun_options(self) -> None:
        """
        Check for the presence of specific srun options.

        Calls `srun --help` and verifying the options. Raises an exception if any required options are missing.
        """
        try:
            result = subprocess.run(["srun", "--help"], text=True, capture_output=True, check=True)
            help_output = result.stdout
        except subprocess.CalledProcessError as e:
            raise EnvironmentError(f"Failed to execute 'srun --help': {e}") from e

        missing_options = [option for option in self.REQUIRED_SRUN_OPTIONS if option not in help_output]
        if missing_options:
            missing_options_str = ", ".join(missing_options)
            raise EnvironmentError(f"Required srun options missing: {missing_options_str}")

    def _write_config(self) -> InstallStatusResult:
        """Write the installation configuration to a TOML file atomically."""
        absolute_install_path = os.path.abspath(self.install_path)
        config_data = {"install_path": absolute_install_path}

        try:
            with open(self.config_path, "w") as file:
                toml.dump(config_data, file)
            return InstallStatusResult(True)
        except Exception as e:
            with contextlib.suppress(OSError):
                os.remove(self.config_path)
            return InstallStatusResult(False, str(e))

    def _read_config(self) -> dict:
        """
        Read the installation configuration from a TOML file.

        Returns
            dict: Configuration, including installation path.
        """
        try:
            with open(self.config_path, "r") as file:
                return toml.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}. "
                "The configuration file is automatically created during installation to store any settings."
            ) from e

    def _remove_config(self) -> None:
        """Remove the installation configuration file."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    def is_installed(self, tests: Iterable[Test]) -> InstallStatusResult:
        """
        Check if the necessary components for the provided test are already installed.

        Verify the existence of the configuration file and the installation status of each test template.

        Args:
            tests (Iterable[Test]): The tests to check for installation.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if not installed.
        """
        if not os.path.exists(self.config_path):
            return InstallStatusResult(
                False,
                f"Configuration file does not exist at {self.config_path}. "
                "The configuration file is automatically created during installation to store any settings.",
            )

        try:
            self._read_config()
        except FileNotFoundError as e:
            return InstallStatusResult(False, str(e))

        return super().is_installed(tests)

    def install(self, tests: Iterable[Test]) -> InstallStatusResult:
        """
        Check if the necessary components are installed and install them if not.

        Requires the installation path to be set.

        Args:
            tests (Iterable[Test]): The tests to install.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        if self.install_path is None:
            return InstallStatusResult(
                False, "Installation path is not set. Please set the install path in the system schema."
            )

        prerequisites_result = self._check_prerequisites()
        if not prerequisites_result:
            return prerequisites_result

        try:
            os.makedirs(self.install_path, exist_ok=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to create installation directory at {self.install_path}: {e}")

        if not os.access(self.install_path, os.W_OK):
            return InstallStatusResult(False, f"The installation path {self.install_path} is not writable.")

        super_result = super().install(tests)
        if not super_result:
            return super_result

        config_result = self._write_config()
        return config_result

    def uninstall(self, tests: Iterable[Test]) -> InstallStatusResult:
        """
        Uninstall the benchmarks or test from the installation path and remove the configuration file.

        This method does not require the installation path to be set in advance.

        Args:
            tests (Iterable[Test]): The tests to uninstall.

        Returns:
            InstallStatusResult: Result containing the uninstallation status and error message if any.
        """
        super_result = super().uninstall(tests)
        if not super_result:
            return super_result

        config = self._read_config()
        install_path = config.get("install_path")
        if install_path:
            try:
                shutil.rmtree(install_path)
            except OSError as e:
                return InstallStatusResult(False, f"Failed to remove installation directory at {install_path}: {e}")
        self._remove_config()
        return InstallStatusResult(True, "All test templates uninstalled successfully.")
