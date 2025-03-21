import logging
import shutil
import subprocess
from pathlib import Path
from shutil import rmtree

from cloudai import BaseInstaller, DockerImage, File, GitRepo, Installable, InstallStatusResult, PythonExecutable
from cloudai.systems import LSFSystem


class LSFInstaller(BaseInstaller):
    """
    Installer for systems that use the LSF scheduler.

    Handles the installation of benchmarks or test templates for LSF-managed systems.

    Attributes:
        PREREQUISITES (List[str]): A list of required binaries for the installer.
        install_path (Path): Path where the benchmarks are to be installed.
    """

    PREREQUISITES = ("bsub", "bjobs", "bhosts", "lsid", "lsload")

    def __init__(self, system: LSFSystem):
        """
        Initialize the LSFInstaller with a system object.

        Args:
            system (LSFSystem): The system schema object.
        """
        super().__init__(system)
        self.system = system

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check for the presence of required binaries, raising an error if any are missing.

        Returns:
            InstallStatusResult: Result containing the status and any error message.
        """
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            return InstallStatusResult(False, base_prerequisites_result.message)

        try:
            self._check_required_binaries()
            return InstallStatusResult(True)
        except EnvironmentError as e:
            return InstallStatusResult(False, str(e))

    def _check_required_binaries(self) -> None:
        """Check for the presence of required binaries, raising an error if any are missing."""
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")

    def install_one(self, item: Installable) -> InstallStatusResult:
        """
        Install a single item.

        Args:
            item (Installable): The item to install.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        logging.debug(f"Attempt to install {item}")

        if isinstance(item, DockerImage):
            logging.info(f"Skipping installation of Docker image {item} in LSF system.")
            return InstallStatusResult(True, "Docker image installation skipped for LSF system.")
        elif isinstance(item, GitRepo):
            return self._install_one_git_repo(item)
        elif isinstance(item, PythonExecutable):
            return self._install_python_executable(item)
        elif isinstance(item, File):
            item.installed_path = self.system.install_path / item.src.name
            shutil.copyfile(item.src, item.installed_path, follow_symlinks=False)
            return InstallStatusResult(True)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        """
        Uninstall a single item.

        Args:
            item (Installable): The item to uninstall.

        Returns:
            InstallStatusResult: Result containing the uninstallation status and error message if any.
        """
        logging.debug(f"Attempt to uninstall {item!r}")
        if isinstance(item, PythonExecutable):
            return self._uninstall_python_executable(item)
        elif isinstance(item, GitRepo):
            return self._uninstall_git_repo(item)
        elif isinstance(item, File):
            if item.installed_path != item.src:
                item.installed_path.unlink()
                item._installed_path = None
                return InstallStatusResult(True)
            logging.debug(f"File {item.installed_path} does not exist.")
            return InstallStatusResult(True)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        """
        Check if a single item is installed.

        Args:
            item (Installable): The item to check.

        Returns:
            InstallStatusResult: Result containing the installation status and error message if any.
        """
        if isinstance(item, DockerImage):
            logging.info(f"Skipping installation check for Docker image {item} in LSF system.")
            return InstallStatusResult(True, "Docker image installation skipped for LSF system.")
        elif isinstance(item, GitRepo):
            repo_path = self.system.install_path / item.repo_name
            if repo_path.exists():
                item.installed_path = repo_path
                return InstallStatusResult(True)
            return InstallStatusResult(False, f"Git repository {item.url} not cloned")
        elif isinstance(item, PythonExecutable):
            return self._is_python_executable_installed(item)
        elif isinstance(item, File):
            if (self.system.install_path / item.src.name).exists():
                item.installed_path = self.system.install_path / item.src.name
                return InstallStatusResult(True)
            return InstallStatusResult(False, f"File {item.installed_path} does not exist")

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        if isinstance(item, GitRepo):
            item.installed_path = self.system.install_path / item.repo_name
            return InstallStatusResult(True)
        elif isinstance(item, PythonExecutable):
            item.git_repo.installed_path = self.system.install_path / item.git_repo.repo_name
            item.venv_path = self.system.install_path / item.venv_name
            return InstallStatusResult(True)
        elif isinstance(item, File):
            item.installed_path = self.system.install_path / item.src.name
            return InstallStatusResult(True)

        return InstallStatusResult(False, f"Unsupported item type: {type(item)}")

    def _install_one_git_repo(self, item: GitRepo) -> InstallStatusResult:
        repo_path = self.system.install_path / item.repo_name
        if repo_path.exists():
            item.installed_path = repo_path
            msg = f"Git repository already exists at {repo_path}."
            logging.warning(msg)
            return InstallStatusResult(True, msg)

        res = self._clone_repository(item.url, repo_path)
        if not res.success:
            return res

        res = self._checkout_commit(item.commit, repo_path)
        if not res.success:
            return res

        item.installed_path = repo_path
        return InstallStatusResult(True)

    def _install_python_executable(self, item: PythonExecutable) -> InstallStatusResult:
        res = self._install_one_git_repo(item.git_repo)
        if not res.success:
            return res

        venv_path = self.system.install_path / item.venv_name
        res = self._create_venv(venv_path)
        if not res.success:
            return res

        assert item.git_repo.installed_path, "Git repository must be installed before creating virtual environment."

        project_dir = item.git_repo.installed_path
        if item.project_subpath:
            project_dir = project_dir / item.project_subpath

        pyproject_toml = project_dir / "pyproject.toml"
        requirements_txt = project_dir / "requirements.txt"

        if pyproject_toml.exists() and requirements_txt.exists():
            if item.dependencies_from_pyproject:
                res = self._install_pyproject(venv_path, project_dir)
            else:
                res = self._install_requirements(venv_path, requirements_txt)
        elif pyproject_toml.exists():
            res = self._install_pyproject(venv_path, project_dir)
        elif requirements_txt.exists():
            res = self._install_requirements(venv_path, requirements_txt)
        else:
            return InstallStatusResult(False, "No pyproject.toml or requirements.txt found for installation.")

        if not res.success:
            return res

        item.venv_path = venv_path
        return InstallStatusResult(True)

    def _clone_repository(self, git_url: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Cloning repository {git_url} into {path}")
        clone_cmd = ["git", "clone", git_url, str(path)]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to clone repository: {result.stderr}")
        return InstallStatusResult(True)

    def _checkout_commit(self, commit_hash: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Checking out specific commit in {path}: {commit_hash}")
        checkout_cmd = ["git", "checkout", commit_hash]
        result = subprocess.run(checkout_cmd, cwd=str(path), capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to checkout commit: {result.stderr}")
        return InstallStatusResult(True)

    def _create_venv(self, venv_dir: Path) -> InstallStatusResult:
        logging.debug(f"Creating virtual environment in {venv_dir}")
        if venv_dir.exists():
            msg = f"Virtual environment already exists at {venv_dir}."
            logging.warning(msg)
            return InstallStatusResult(True, msg)

        result = subprocess.run(["python", "-m", "venv", str(venv_dir)], capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to create venv: {result.stderr}")
        return InstallStatusResult(True)

    def _install_pyproject(self, venv_dir: Path, project_dir: Path) -> InstallStatusResult:
        install_cmd = [str(venv_dir / "bin" / "python"), "-m", "pip", "install", str(project_dir)]
        result = subprocess.run(install_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install {project_dir} using pip: {result.stderr}")

        return InstallStatusResult(True)

    def _install_requirements(self, venv_dir: Path, requirements_txt: Path) -> InstallStatusResult:
        if not requirements_txt.is_file():
            return InstallStatusResult(False, f"Requirements file is invalid or does not exist: {requirements_txt}")

        install_cmd = [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_txt)]
        result = subprocess.run(install_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install dependencies from requirements.txt: {result.stderr}")

        return InstallStatusResult(True)

    def _uninstall_git_repo(self, item: GitRepo) -> InstallStatusResult:
        logging.debug(f"Uninstalling git repository at {item.installed_path=}")
        repo_path = item.installed_path if item.installed_path else self.system.install_path / item.repo_name
        if not repo_path.exists():
            msg = f"Repository {item.url} is not cloned."
            return InstallStatusResult(True, msg)

        logging.debug(f"Removing folder {repo_path}")
        rmtree(repo_path)
        item.installed_path = None

        return InstallStatusResult(True)

    def _uninstall_python_executable(self, item: PythonExecutable) -> InstallStatusResult:
        res = self._uninstall_git_repo(item.git_repo)
        if not res.success:
            return res

        logging.debug(f"Uninstalling virtual environment at {item.venv_path=}")
        venv_path = item.venv_path if item.venv_path else self.system.install_path / item.venv_name
        if not venv_path.exists():
            msg = f"Virtual environment {item.venv_name} is not created."
            return InstallStatusResult(True, msg)

        logging.debug(f"Removing folder {venv_path}")
        rmtree(venv_path)
        item.venv_path = None

        return InstallStatusResult(True)

    def _is_python_executable_installed(self, item: PythonExecutable) -> InstallStatusResult:
        repo_path = (
            item.git_repo.installed_path
            if item.git_repo.installed_path
            else self.system.install_path / item.git_repo.repo_name
        )
        if not repo_path.exists():
            return InstallStatusResult(False, f"Git repository {item.git_repo.url} not cloned")
        item.git_repo.installed_path = repo_path

        venv_path = item.venv_path if item.venv_path else self.system.install_path / item.venv_name
        if not venv_path.exists():
            return InstallStatusResult(False, f"Virtual environment not created for {item.git_repo.url}")
        item.venv_path = venv_path

        return InstallStatusResult(True, "Python executable installed")