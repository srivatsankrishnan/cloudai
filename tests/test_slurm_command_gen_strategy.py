# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from cloudai.schema.test_template.jax_toolbox.slurm_command_gen_strategy import JaxToolboxSlurmCommandGenStrategy
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.schema.test_template.nemo_launcher.slurm_command_gen_strategy import (
    NeMoLauncherSlurmCommandGenStrategy,
)
from cloudai.systems import SlurmSystem
from cloudai.systems.slurm import SlurmNodeState
from cloudai.systems.slurm.slurm_system import SlurmPartition
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.test_definitions.gpt import GPTCmdArgs, GPTTestDefinition
from cloudai.test_definitions.grok import GrokCmdArgs, GrokTestDefinition
from cloudai.test_definitions.jax_toolbox import PreTest


@pytest.fixture
def slurm_system(tmp_path: Path) -> SlurmSystem:
    slurm_system = SlurmSystem(
        name="TestSystem",
        install_path=tmp_path / "install",
        output_path=tmp_path / "output",
        default_partition="main",
        partitions=[
            SlurmPartition(name="main", nodes=["node[1-4]"]),
        ],
        mpi="fake-mpi",
    )
    for node in slurm_system.partitions[0].slurm_nodes:
        node.state = SlurmNodeState.IDLE
    Path(slurm_system.install_path).mkdir()
    Path(slurm_system.output_path).mkdir()
    return slurm_system


@pytest.fixture
def strategy_fixture(slurm_system: SlurmSystem) -> SlurmCommandGenStrategy:
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    strategy = SlurmCommandGenStrategy(slurm_system, env_vars, cmd_args)
    return strategy


@pytest.fixture
def jax_strategy_fixture() -> JaxToolboxSlurmCommandGenStrategy:
    # Mock the SlurmSystem and other dependencies
    mock_slurm_system = Mock()
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    mock_slurm_system.install_path = "/mock/install/path"

    # Use patch to mock the __init__ method of JaxToolboxSlurmCommandGenStrategy
    with patch.object(JaxToolboxSlurmCommandGenStrategy, "__init__", lambda self, _, __, ___: None):
        strategy = JaxToolboxSlurmCommandGenStrategy(mock_slurm_system, env_vars, cmd_args)
        # Manually set attributes needed for the tests
        strategy.env_vars = env_vars
        strategy.cmd_args = cmd_args
        strategy.default_env_vars = env_vars
        strategy.default_cmd_args = cmd_args
        return strategy


@pytest.fixture
def gpt_test() -> GPTTestDefinition:
    return GPTTestDefinition(
        name="gpt",
        description="desc",
        test_template_name="gpt",
        cmd_args=GPTCmdArgs(fdl_config=""),
        extra_env_vars={"COMBINE_THRESHOLD": "1"},  # it is always set in Test TOMLs
    )


@pytest.fixture
def grok_test() -> GrokTestDefinition:
    return GrokTestDefinition(
        name="grok",
        description="desc",
        test_template_name="grok",
        cmd_args=GrokCmdArgs(),
        extra_env_vars={"COMBINE_THRESHOLD": "1"},  # it is always set in Test TOMLs
    )


def test_filename_generation(strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
    args = {"job_name": "test_job", "num_nodes": 2, "partition": "test_partition", "node_list_str": "node1,node2"}
    env_vars_str = "export TEST_VAR=VALUE"
    srun_command = "srun --test test_arg"
    output_path = tmp_path

    sbatch_command = strategy_fixture._write_sbatch_script(args, env_vars_str, srun_command, output_path)
    filepath_from_command = sbatch_command.split()[-1]

    # Check that the file exists at the specified path
    assert output_path.joinpath("cloudai_sbatch_script.sh").exists()

    # Read the file and check the contents
    with open(filepath_from_command, "r") as file:
        file_contents = file.read()
    assert "test_job" in file_contents
    assert "node1,node2" in file_contents
    assert "srun --test test_arg" in file_contents

    # Check the correctness of the sbatch command format
    assert sbatch_command == f"sbatch {filepath_from_command}"


def test_num_nodes_and_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


def test_only_num_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = []
    num_nodes = 3

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == num_nodes


def test_only_nodes(strategy_fixture: SlurmCommandGenStrategy):
    job_name_prefix = "test_job"
    env_vars = {"TEST_VAR": "VALUE"}
    cmd_args = {"test_arg": "test_value"}
    nodes = ["node1", "node2"]
    num_nodes = 0

    slurm_args = strategy_fixture._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

    assert slurm_args["num_nodes"] == len(nodes)


def test_raises_if_no_default_partition(slurm_system: SlurmSystem):
    slurm_system.default_partition = ""
    with pytest.raises(ValueError) as exc_info:
        SlurmCommandGenStrategy(slurm_system, {}, {})
    assert "Partition not specified in the system configuration." in str(exc_info)


class TestGenerateSrunCommand__CmdGeneration:
    def test_generate_test_command(self, strategy_fixture: SlurmCommandGenStrategy):
        test_command = strategy_fixture.generate_test_command({}, {}, {}, "")
        assert test_command == []

    def test_generate_srun_command(self, strategy_fixture: SlurmCommandGenStrategy):
        srun_command = strategy_fixture.generate_srun_command({}, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}"]

    def test_generate_srun_command_with_extra_args(self, strategy_fixture: SlurmCommandGenStrategy):
        strategy_fixture.slurm_system.extra_srun_args = "--extra-args value"
        srun_command = strategy_fixture.generate_srun_command({}, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}", "--extra-args value"]

    def test_generate_srun_command_with_container_image(self, strategy_fixture: SlurmCommandGenStrategy):
        slurm_args = {"image_path": "fake_image_path"}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == [
            "srun",
            f"--mpi={strategy_fixture.slurm_system.mpi}",
            "--container-image=fake_image_path",
        ]

    def test_generate_srun_command_with_container_image_and_mounts(self, strategy_fixture: SlurmCommandGenStrategy):
        slurm_args = {"image_path": "fake_image_path", "container_mounts": "fake_mounts"}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == [
            "srun",
            f"--mpi={strategy_fixture.slurm_system.mpi}",
            "--container-image=fake_image_path",
            "--container-mounts=fake_mounts",
        ]

    def test_generate_srun_empty_str(self, strategy_fixture: SlurmCommandGenStrategy):
        slurm_args = {"image_path": "", "container_mounts": ""}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}"]

        slurm_args = {"image_path": "fake", "container_mounts": ""}
        srun_command = strategy_fixture.generate_srun_command(slurm_args, {}, {}, "")
        assert srun_command == ["srun", f"--mpi={strategy_fixture.slurm_system.mpi}", "--container-image=fake"]

    def test_generate_full_srun_command(self, strategy_fixture: SlurmCommandGenStrategy):
        strategy_fixture.generate_srun_command = lambda *_, **__: ["srun", "--test", "test_arg"]
        strategy_fixture.generate_test_command = lambda *_, **__: ["test_command"]

        full_srun_command = strategy_fixture.generate_full_srun_command({}, {}, {}, "")
        assert full_srun_command == " \\\n".join(["srun", "--test", "test_arg", "test_command"])

    def test_generate_full_srun_command_with_pre_test(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.extra_env_vars, gpt_test.cmd_args_dict)
        cmd_gen._create_run_script = MagicMock()
        cmd_gen._generate_pre_test_command = MagicMock(return_value="pre_test_command")
        cmd_gen._generate_pre_test_check_command = MagicMock(return_value="pre_test_check_command")

        slurm_args = {
            "output": "output.txt",
            "error": "error.txt",
            "image_path": "image_path",
            "container_mounts": "container_mounts",
            "container_name": "cont",
        }
        gpt_test.cmd_args.pre_test = PreTest()
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        result = cmd_gen.generate_full_srun_command(slurm_args, {}, cargs, "")
        assert "pre_test_command" in result
        assert "pre_test_check_command" in result
        assert "--mpi=none" in result
        assert "--container-mounts=" + slurm_args["container_mounts"] in result
        assert f"-o {slurm_args['output']}" in result
        assert f"-e {slurm_args['error']}" in result
        assert "--container-name=" + slurm_args["container_name"] in result
        assert "/opt/paxml/workspace/run.sh" in result

    def test_generate_full_srun_command_without_pre_test(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.extra_env_vars, gpt_test.cmd_args_dict)
        cmd_gen._create_run_script = MagicMock()
        cmd_gen._generate_pre_test_command = MagicMock(return_value="pre_test_command")
        cmd_gen._generate_pre_test_check_command = MagicMock(return_value="pre_test_check_command")

        slurm_args = {
            "output": "output.txt",
            "error": "error.txt",
            "image_path": "image_path",
            "container_mounts": "container_mounts",
            "container_name": "cont",
        }
        gpt_test.cmd_args.pre_test = PreTest(enable=False)
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        result = cmd_gen.generate_full_srun_command(slurm_args, {}, cargs, "")

        assert "pre_test_command" not in result
        assert "pre_test_check_command" not in result
        assert "--mpi=none" in result
        assert f"--container-mounts={slurm_args['container_mounts']}" in result
        assert "--container-name=" + slurm_args.get("container_name", "") in result
        assert f"-o {slurm_args['output']}" in result
        assert f"-e {slurm_args['error']}" in result

    def test_gen_exec_command(self, slurm_system: SlurmSystem, tmp_path: Path, gpt_test: GPTTestDefinition):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.extra_env_vars, gpt_test.cmd_args_dict)
        cmd = cmd_gen.gen_exec_command({}, gpt_test.cmd_args_dict, gpt_test.extra_env_vars, "", tmp_path, 1, ["node1"])
        assert cmd == f"sbatch {tmp_path}/cloudai_sbatch_script.sh"
        assert (tmp_path / "run.sh").exists()

    def test_generate_pre_test_command(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition, tmp_path: Path):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.extra_env_vars, gpt_test.cmd_args_dict)
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        pre_test_cli = cmd_gen._generate_pre_test_command(cargs, tmp_path, tmp_path).splitlines()
        nccl_test = gpt_test.cmd_args.pre_test.nccl_test
        assert pre_test_cli == [
            "srun \\",
            "--mpi=pmix \\",
            f"-N {nccl_test.num_nodes} \\",
            f"-o {tmp_path} \\",
            f"-e {tmp_path} \\",
            f"--container-image={nccl_test.docker_image_url} \\",
            f"/usr/local/bin/{nccl_test.subtest_name} \\",
            f"--nthreads {nccl_test.nthreads} \\",
            f"--ngpus {nccl_test.ngpus} \\",
            f"--minbytes {nccl_test.minbytes} \\",
            f"--maxbytes {nccl_test.maxbytes} \\",
            f"--stepbytes {nccl_test.stepbytes} \\",
            f"--op {nccl_test.op} \\",
            f"--datatype {nccl_test.datatype} \\",
            f"--root {nccl_test.root} \\",
            f"--iters {nccl_test.iters} \\",
            f"--warmup_iters {nccl_test.warmup_iters} \\",
            f"--agg_iters {nccl_test.agg_iters} \\",
            f"--average {nccl_test.average} \\",
            f"--parallel_init {nccl_test.parallel_init} \\",
            f"--check {nccl_test.check} \\",
            f"--blocking {nccl_test.blocking} \\",
            f"--cudagraph {nccl_test.cudagraph} \\",
            f"--stepfactor {nccl_test.stepfactor}",
        ]

    def test_generate_python_command(self, slurm_system: SlurmSystem, gpt_test: GPTTestDefinition, tmp_path: Path):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.extra_env_vars, gpt_test.cmd_args_dict)
        cargs = {"output_path": "/path/to/output", **gpt_test.cmd_args_dict}
        cmd_gen.test_name = "GPT"
        stage = "stage"
        python_cli = cmd_gen._generate_python_command(stage, {}, {}, cargs, "").splitlines()

        fdl_args = gpt_test.cmd_args.fdl.model_dump()
        fdl_args_list = []
        for k, v in sorted(fdl_args.items()):
            fdl_args_list.append(f"    --fdl.{k.upper()}={v} \\")
        fdl_args_list[-1] = fdl_args_list[-1].replace(" \\", "")
        py_cmd = [
            "    python3 -u -m paxml.main \\",
            "    --num_hosts=$SLURM_NTASKS \\",
            "    --server_addr=$SLURM_JOB_MASTER_NODE:12345 \\",
            "    --host_idx=$SLURM_PROCID \\",
            f"    --job_log_dir={gpt_test.cmd_args.setup_flags.docker_workspace_dir} \\",
            f"    --tfds_data_dir={gpt_test.cmd_args.setup_flags.tfds_data_dir} \\",
            f"    --enable_checkpoint_saving={gpt_test.cmd_args.setup_flags.enable_checkpoint_saving} \\",
            "    --multiprocess_gpu \\",
            "    --alsologtostderr \\",
            f'    --fdl_config="{gpt_test.cmd_args.fdl_config}" \\',
            *fdl_args_list,
        ]

        assert python_cli == [
            'if [ "$SLURM_NODEID" -eq 0 ] && [ "$SLURM_PROCID" -eq 0 ]; then',
            "    nsys profile \\",
            "    -s none \\",
            f"    -o /opt/paxml/workspace/nsys_profile_{stage} \\",
            "    --force-overwrite true \\",
            "    --capture-range=cudaProfilerApi \\",
            "    --capture-range-end=stop \\",
            "    --cuda-graph-trace=node \\",
            *py_cmd,
            "else",
            *py_cmd,
            "fi",
        ]

    @pytest.mark.parametrize("enable_pgle,expected_ncalls", [(True, 2), (False, 1)])
    def test_create_run_script_pgle_control(
        self,
        enable_pgle: bool,
        expected_ncalls: int,
        slurm_system: SlurmSystem,
        grok_test: GrokTestDefinition,
        tmp_path: Path,
    ):
        grok_test.cmd_args.enable_pgle = enable_pgle
        cmd_args = grok_test.cmd_args_dict
        cmd_args["output_path"] = str(tmp_path)
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, grok_test.extra_env_vars, cmd_args)
        cmd_gen.test_name = "Grok"
        cmd_gen._script_content = Mock(return_value="")
        cmd_gen._create_run_script({}, grok_test.extra_env_vars, cmd_args, "")
        assert (tmp_path / "run.sh").exists()
        assert cmd_gen._script_content.call_count == expected_ncalls


class TestJaxToolboxSlurmCommandGenStrategy__ExtractTestName:
    @pytest.mark.parametrize(
        "cmd_args, expected",
        [
            ({"Grok.setup_flags": "/some/dir"}, "Grok"),
            ({"GPT.setup_flags": "/some/dir"}, "GPT"),
            ({"Nemotron.setup_flags": "/some/dir"}, "Nemotron"),
            ({"unknown": "value"}, ""),
        ],
    )
    def test_extract_test_name(
        self, cmd_args: dict, expected: str, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy
    ):
        test_name = jax_strategy_fixture._extract_test_name(cmd_args)
        assert test_name == expected

    def test_format_xla_flags_grok(self, grok_test: GrokTestDefinition, slurm_system: SlurmSystem):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, grok_test.extra_env_vars, grok_test.cmd_args_dict)
        cmd_gen.test_name = "Grok"

        xla_flags = cmd_gen._format_xla_flags(grok_test.cmd_args_dict, "profile")

        actual_flags_list = sorted(xla_flags.split())

        profile_xlas = [f"--{k}={v}" for k, v in grok_test.cmd_args.profile.model_dump().items()]
        profile_xlas = [x.replace("=False", "=false") for x in profile_xlas]
        profile_xlas = [x.replace("=True", "=true") for x in profile_xlas]
        expected_flags_list = sorted(
            [
                "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
                *profile_xlas,
            ]
        )
        assert actual_flags_list == expected_flags_list

        xla_flags = cmd_gen._format_xla_flags(grok_test.cmd_args_dict, "perf")

        actual_flags_list = sorted(xla_flags.split())

        perf_xlas = [f"--{k}={v}" for k, v in grok_test.cmd_args.perf.model_dump().items() if k.startswith("xla_")]
        perf_xlas = [x.replace("=False", "=false") for x in perf_xlas]
        perf_xlas = [x.replace("=True", "=true") for x in perf_xlas]
        expected_flags_list = sorted(
            [
                "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD",
                "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD",
                *perf_xlas,
            ]
        )
        assert actual_flags_list == expected_flags_list

    def test_format_xla_flags_gpt(self, gpt_test: GPTTestDefinition, slurm_system: SlurmSystem):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, gpt_test.extra_env_vars, gpt_test.cmd_args_dict)
        cmd_gen.test_name = "GPT"
        xla_flags = cmd_gen._format_xla_flags(gpt_test.cmd_args_dict, "profile")

        actual_flags_list = sorted(xla_flags.split())
        expected_flags_list = sorted(
            "--xla_gpu_all_reduce_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_all_gather_combine_threshold_bytes=$COMBINE_THRESHOLD "
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=$PER_GPU_COMBINE_THRESHOLD".split()
        )
        assert actual_flags_list == expected_flags_list

    def test_format_xla_flags_boolean_are_lowcased(self, slurm_system: SlurmSystem):
        cmd_gen = JaxToolboxSlurmCommandGenStrategy(slurm_system, {}, {})
        cmd_gen.test_name = "GPT"

        xla_flags = cmd_gen._format_xla_flags(
            {"GPT.profile": {"XLA_FLAGS": {"xla_gpu_enable_while_loop_double_buffering": True}}}, "profile"
        ).split()
        assert len(xla_flags) == 4
        assert xla_flags[-2] == "--xla_gpu_enable_while_loop_double_buffering=true"

    def test_handle_threshold_and_env_common(self, jax_strategy_fixture: JaxToolboxSlurmCommandGenStrategy):
        cmd_args = {"GPT.XLA_FLAGS": "combine_threshold_bytes", "GPT.setup_flags": {"gpus_per_node": 4}}
        env_vars = {}
        combine_threshold_bytes = 1024
        num_nodes = 2

        jax_strategy_fixture.env_vars = env_vars
        jax_strategy_fixture.cmd_args = cmd_args
        jax_strategy_fixture.test_name = "GPT"

        jax_strategy_fixture._handle_threshold_and_env(cmd_args, env_vars, combine_threshold_bytes, num_nodes)

        assert "PER_GPU_COMBINE_THRESHOLD" in env_vars
        assert env_vars["PER_GPU_COMBINE_THRESHOLD"] == str(
            combine_threshold_bytes // (cmd_args["GPT.setup_flags"]["gpus_per_node"] * num_nodes)
        )
        assert "XLA_FLAGS.combine_threshold_bytes" not in cmd_args


class TestNeMoLauncherSlurmCommandGenStrategy__GenExecCommand:
    @pytest.fixture
    def nemo_cmd_gen(self, slurm_system: SlurmSystem) -> NeMoLauncherSlurmCommandGenStrategy:
        env_vars = {"TEST_VAR": "VALUE"}
        cmd_args = {"test_arg": "test_value"}
        strategy = NeMoLauncherSlurmCommandGenStrategy(slurm_system, env_vars, cmd_args)
        return strategy

    def test_extra_env_vars_added(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR_1": "value1", "TEST_VAR_2": "value2"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args="",
            output_path=Path(""),
            num_nodes=1,
            nodes=[],
        )

        for k, v in extra_env_vars.items():
            assert f"{k}={v}" in cmd

    def test_env_var_escaping(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR": "value,with,commas"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args="",
            output_path=Path(""),
            num_nodes=1,
            nodes=[],
        )

        assert "TEST_VAR=\\'value,with,commas\\'" in cmd

    def test_tokenizer_handled(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy, tmp_path: Path):
        extra_env_vars = {"TEST_VAR_1": "value1"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        tokenizer_path = tmp_path / "tokenizer"
        tokenizer_path.touch()

        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_env_vars=extra_env_vars,
            extra_cmd_args=f"training.model.tokenizer.model={tokenizer_path}",
            output_path=Path(""),
            num_nodes=1,
            nodes=[],
        )

        assert f"container_mounts=[{tokenizer_path}:{tokenizer_path}]" in cmd

    def test_reservation_handled(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR_1": "value1"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        nemo_cmd_gen.slurm_system.extra_srun_args = "--reservation my-reservation"
        cmd = nemo_cmd_gen.gen_exec_command(
            env_vars={},
            cmd_args=cmd_args,
            extra_cmd_args="",
            extra_env_vars=extra_env_vars,
            output_path=Path(""),
            num_nodes=1,
            nodes=[],
        )

        assert "+cluster.reservation=my-reservation" in cmd

    def test_invalid_tokenizer_path(self, nemo_cmd_gen: NeMoLauncherSlurmCommandGenStrategy):
        extra_env_vars = {"TEST_VAR_1": "value1"}
        cmd_args = {
            "docker_image_url": "fake",
            "repository_url": "fake",
            "repository_commit_hash": "fake",
        }
        invalid_tokenizer_path = Path("/invalid/path/to/tokenizer")

        with pytest.raises(
            ValueError,
            match=(
                r"The provided tokenizer path '/invalid/path/to/tokenizer' is not valid. Please review the test "
                r"schema file to ensure the tokenizer path is correct. If it contains a placeholder value, refer to "
                r"USER_GUIDE.md to download the tokenizer and update the schema file accordingly."
            ),
        ):
            nemo_cmd_gen.gen_exec_command(
                env_vars={},
                cmd_args=cmd_args,
                extra_env_vars=extra_env_vars,
                extra_cmd_args=f"training.model.tokenizer.model={invalid_tokenizer_path}",
                output_path=Path(""),
                num_nodes=1,
                nodes=[],
            )


class TestWriteSbatchScript:
    MANDATORY_ARGS = {
        "job_name": "test_job",
        "num_nodes": 2,
        "node_list_str": "node1,node2",
    }

    def setup_method(self):
        self.env_vars_str = "export TEST_VAR=VALUE"
        self.srun_command = "srun --test test_arg"

    def assert_slurm_directives(self, lines: list[str]):
        assert lines[0] == "#!/bin/bash"

        assert f"#SBATCH --job-name={self.MANDATORY_ARGS['job_name']}" in lines
        assert f"#SBATCH -N {self.MANDATORY_ARGS['num_nodes']}" in lines

        partition = self.MANDATORY_ARGS.get("partition")
        if partition:
            assert f"#SBATCH --partition={partition}" in lines

        node_list_str = self.MANDATORY_ARGS.get("node_list_str")
        if node_list_str:
            assert f"#SBATCH --nodelist={node_list_str}" in lines

        gpus_per_node = self.MANDATORY_ARGS.get("gpus_per_node")
        if gpus_per_node:
            assert f"#SBATCH --gpus-per-node={gpus_per_node}" in lines
            assert f"#SBATCH --gres=gpu:{gpus_per_node}" in lines

        ntasks_per_node = self.MANDATORY_ARGS.get("ntasks_per_node")
        if ntasks_per_node:
            assert f"#SBATCH --ntasks-per-node={ntasks_per_node}" in lines

        time_limit = self.MANDATORY_ARGS.get("time_limit")
        if time_limit:
            assert f"#SBATCH --time={time_limit}" in lines

    @pytest.mark.parametrize("missing_arg", ["job_name", "num_nodes"])
    def test_raises_on_missing_args(self, missing_arg: str, strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
        args = self.MANDATORY_ARGS.copy()
        del args[missing_arg]

        with pytest.raises(KeyError) as exc_info:
            strategy_fixture._write_sbatch_script(args, self.env_vars_str, self.srun_command, tmp_path)
        assert missing_arg in str(exc_info.value)

    def test_only_mandatory_args(self, strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
        sbatch_command = strategy_fixture._write_sbatch_script(
            self.MANDATORY_ARGS, self.env_vars_str, self.srun_command, tmp_path
        )

        filepath_from_command = sbatch_command.split()[-1]
        assert sbatch_command == f"sbatch {filepath_from_command}"
        assert tmp_path.joinpath("cloudai_sbatch_script.sh").exists()

        with open(filepath_from_command, "r") as file:
            file_contents = file.read()

        lines = file_contents.splitlines()

        assert len(lines) == 12

        self.assert_slurm_directives(lines)

        # Check for the specific lines in the file
        assert f"#SBATCH --job-name={self.MANDATORY_ARGS['job_name']}" in file_contents
        assert f"#SBATCH -N {self.MANDATORY_ARGS['num_nodes']}" in file_contents
        assert f"#SBATCH --partition={strategy_fixture.slurm_system.default_partition}" in file_contents
        assert f"#SBATCH --nodelist={self.MANDATORY_ARGS['node_list_str']}" in file_contents
        assert f"#SBATCH --output={tmp_path / 'stdout.txt'}" in file_contents
        assert f"#SBATCH --error={tmp_path / 'stderr.txt'}" in file_contents

    @pytest.mark.parametrize(
        "arg, arg_value, expected_str",
        [
            ("account", "test_account", None),
            ("distribution", "block", None),
            ("gpus_per_node", 2, None),
            ("ntasks_per_node", 2, None),
            ("time_limit", "00:30:00", "#SBATCH --time=00:30:00"),
        ],
    )
    def test_extra_args(
        self,
        arg: str,
        arg_value: str,
        expected_str: Optional[str],
        strategy_fixture: SlurmCommandGenStrategy,
        tmp_path: Path,
    ):
        args = self.MANDATORY_ARGS.copy()
        if expected_str:  # use slurm_args
            args[arg] = arg_value
        else:  # use strategy.slurm_system.<arg>
            v = getattr(strategy_fixture.slurm_system, arg)
            if not v:
                setattr(strategy_fixture.slurm_system, arg, arg_value)
                v = arg_value
            str_arg = arg.replace("_", "-")
            expected_str = f"#SBATCH --{str_arg}={v}"

        sbatch_command = strategy_fixture._write_sbatch_script(args, self.env_vars_str, self.srun_command, tmp_path)

        filepath_from_command = sbatch_command.split()[-1]
        with open(filepath_from_command, "r") as file:
            file_contents = file.read()

        self.assert_slurm_directives(file_contents.splitlines())
        assert expected_str in file_contents

    def test_reservation(
        self,
        strategy_fixture: SlurmCommandGenStrategy,
        tmp_path: Path,
    ):
        strategy_fixture.slurm_system.extra_srun_args = "--reservation my-reservation"
        args = self.MANDATORY_ARGS.copy()

        sbatch_command = strategy_fixture._write_sbatch_script(args, self.env_vars_str, self.srun_command, tmp_path)
        filepath_from_command = sbatch_command.split()[-1]
        with open(filepath_from_command, "r") as file:
            file_contents = file.read()

        assert "#SBATCH --reservation=my-reservation" in file_contents

    @pytest.mark.parametrize("add_arg", ["output", "error"])
    def test_disable_output_and_error(self, add_arg: str, strategy_fixture: SlurmCommandGenStrategy, tmp_path: Path):
        args = self.MANDATORY_ARGS.copy()
        args[add_arg] = "fake"

        sbatch_command = strategy_fixture._write_sbatch_script(args, self.env_vars_str, self.srun_command, tmp_path)

        filepath_from_command = sbatch_command.split()[-1]
        with open(filepath_from_command, "r") as file:
            file_contents = file.read()

        self.assert_slurm_directives(file_contents.splitlines())
        assert f"--{add_arg}=" not in file_contents


class TestNCCLSlurmCommandGen:
    def get_cmd(self, slurm_system: SlurmSystem, slurm_args: dict, cmd_args: dict) -> str:
        return NcclTestSlurmCommandGenStrategy(slurm_system, {}, {}).generate_full_srun_command(
            slurm_args, {}, cmd_args, ""
        )

    def test_only_mandatory(self, slurm_system: SlurmSystem) -> None:
        slurm_args = {"image_path": "fake_image_path"}
        cmd_args = {"subtest_name": "fake_subtest_name"}
        cmd = self.get_cmd(slurm_system, slurm_args, cmd_args)
        assert cmd == " \\\n".join(
            [
                "srun",
                f"--mpi={slurm_system.mpi}",
                f"--container-image={slurm_args['image_path']}",
                f"/usr/local/bin/{cmd_args['subtest_name']}",
            ]
        )

    def test_with_container_mounts(self, slurm_system: SlurmSystem) -> None:
        slurm_args = {"image_path": "fake_image_path", "container_mounts": "fake_mounts"}
        cmd_args = {"subtest_name": "fake_subtest_name"}
        cmd = self.get_cmd(slurm_system, slurm_args, cmd_args)
        assert cmd == " \\\n".join(
            [
                "srun",
                f"--mpi={slurm_system.mpi}",
                f"--container-image={slurm_args['image_path']}",
                f"--container-mounts={slurm_args['container_mounts']}",
                f"/usr/local/bin/{cmd_args['subtest_name']}",
            ]
        )
