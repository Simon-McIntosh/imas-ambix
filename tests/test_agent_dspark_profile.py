"""Profile-to-launch contracts for the DSpark four-card DeepSeek-V4-Flash serve."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from click.testing import CliRunner

from imas_ambix.agent.profile import ModelProfile, SiteConfig, load_profile
from imas_ambix.agent.slurm import generate_download_script, generate_serve_script
from imas_ambix.cli import main

CHECKPOINT_SHA = "1bae0e1088993b9b82de08f59de5be5a34e42170"
CHECKPOINT_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
CHECKPOINT_MODEL_DIR = "/work/projects/imas_gpu/agents/deepseek-v4-1-flash-1017/model"
SEPTEMBER_MODEL_DIR = "/work/projects/imas_gpu/agents/deepseek-v4-1-flash/model"


def test_profile_pins_the_checkpoint_it_serves() -> None:
    """The profile names the release, not the repository family it lives in.

    The 1017 update publishes weights at a commit of that repository so its
    repo id alone would leave a download free to follow the default branch.
    weights_slug keeps one release per directory, and served_name stays the
    single client-visible name every release of this family answers to.
    """
    profile = load_profile("deepseek-v4-1-flash")

    assert profile.model.hf_repo == CHECKPOINT_REPO
    assert profile.model.revision == CHECKPOINT_SHA
    assert profile.weights_directory_slug == "deepseek-v4-1-flash-1017"
    assert profile.model.served_name == "deepseek-v4.1-flash"


def test_serve_script_moves_only_the_weights_directory() -> None:
    """The release change reaches the serve as the directory it loads from.

    The baseline is this same profile with the September weights directory
    restored, so what is asserted is that the release change moves exactly one
    line of the generated script -- a change that moved anything else would be
    a serving-setting change dressed as a checkpoint pin.
    """
    profile = load_profile("deepseek-v4-1-flash")
    september = profile.model_copy(deep=True)
    september.model.weights_slug = "deepseek-v4-1-flash"

    pinned = generate_serve_script(profile, SiteConfig(), port=18810)
    baseline = generate_serve_script(september, SiteConfig(), port=18810)

    assert f"MODEL_DIR={CHECKPOINT_MODEL_DIR}" in pinned
    assert len(pinned.splitlines()) == len(baseline.splitlines())
    differing = [
        (was, now)
        for was, now in zip(baseline.splitlines(), pinned.splitlines(), strict=True)
        if was != now
    ]
    assert differing == [
        (
            "        MODEL_DIR=" + SEPTEMBER_MODEL_DIR,
            f"        MODEL_DIR={CHECKPOINT_MODEL_DIR}",
        )
    ]


def _speculative_config(script: str) -> dict[str, object]:
    line = next(
        line
        for line in script.splitlines()
        if "vllm.entrypoints.openai.api_server" in line
    )
    tokens = shlex.split(line)
    value = tokens[tokens.index("--speculative-config") + 1]
    return json.loads(value)


def test_four_card_profile_declares_dspark_engine_flags() -> None:
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    assert profile.engine.type == "vllm"
    assert profile.engine.block_size == 256
    assert profile.engine.tokenizer_mode == "deepseek_v4"
    assert profile.engine.moe_backend == "auto"
    assert profile.engine.enable_expert_parallel is True
    assert profile.engine.speculative_method == "dspark"
    assert profile.engine.speculative_num_tokens == 5
    assert profile.engine.speculative_draft_sample_method == "greedy"


def test_four_card_serve_script_emits_dspark_recipe_flags() -> None:
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    script = generate_serve_script(profile, SiteConfig(), port=18800)

    assert "--block-size 256" in script
    assert "--enable-expert-parallel" in script
    assert "--tokenizer-mode deepseek_v4" in script
    assert "--moe-backend auto" in script
    assert "--speculative-config" in script
    assert _speculative_config(script) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "greedy",
    }
    # DSpark's module is fused into the checkpoint -- no separate draft model.
    assert "model" not in _speculative_config(script)


def test_four_card_gpu_variant_selection_matches_gpus_flag() -> None:
    """Card count is tensor width times replica count, not tensor width alone.

    Four cards carry two two-wide replicas rather than one four-wide engine,
    so asserting a tensor width of four would pin one topology rather than the
    invariant that makes a variant selection correct. The product is what must
    equal the cards the scheduler is asked for; a variant whose product misses
    the card count either strands a device or over-requests one.
    """
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    assert profile.slurm.gpus == 4
    assert profile.engine.tensor_parallel * profile.engine.data_parallel == 4


def test_every_gpu_variant_topology_fills_its_card_count() -> None:
    """The product invariant holds for every declared variant, not just four.

    The eight-card entry sets a tensor width of four precisely because the
    replica count carries the other factor; carrying the card count straight
    into tensor_parallel there would request sixteen devices.
    """
    base = load_profile("deepseek-v4-flash")

    for gpus in sorted({2, 4, 8} | set(base.gpu_variants)):
        profile = base.for_gpus(gpus)
        product = profile.engine.tensor_parallel * profile.engine.data_parallel
        assert profile.slurm.gpus == gpus, (
            f"{gpus}-card variant declares {profile.slurm.gpus}"
        )
        assert product == gpus, (
            f"{gpus}-card variant spreads over {product} devices "
            f"(tp={profile.engine.tensor_parallel}, dp={profile.engine.data_parallel})"
        )


def test_four_card_serve_script_carries_full_native_context_window() -> None:
    """The four-card serve sets the checkpoint's native 1M context at a memory
    fraction that fits its measured KV pool. The single compressed KV head
    does not shard across tensor parallelism, so the four-card pool matches
    the two-card one.
    """
    profile = load_profile("deepseek-v4-flash").for_gpus(4)

    script = generate_serve_script(profile, SiteConfig(), port=18800)

    assert "--max-model-len 1048576" in script
    assert "--gpu-memory-utilization 0.92" in script


def test_speculative_config_dotted_form_survives_when_no_draft_sample_method() -> None:
    """The GLM-5.2 MTP path (no draft model, no draft-sample method) keeps the
    dotted ``--speculative-config.*`` flags rather than the compact JSON form.
    """
    profile = load_profile("glm-5-2")

    script = generate_serve_script(profile, SiteConfig(), port=18800)

    assert "--speculative-config.method mtp" in script
    assert "--speculative-config.num_speculative_tokens 5" in script
    assert "--speculative-config '" not in script


def test_no_profile_declares_the_router_port():
    """No serving profile may claim the port the router runs on.

    A profile declaring the router's port makes a bare ``agent serve <slug>``
    refuse at submit with a port-holder error naming ``ambix-router``, which
    reads as a router fault rather than as a profile defect. The two-card
    DeepSeek variant declared 18802 until 2026-09-06.
    """
    from imas_ambix.agent.profile import list_profiles, load_profile

    router_port = 18802
    for slug in list_profiles():
        profile = load_profile(slug)
        assert profile.slurm.port != router_port, (
            f"profile {slug!r} declares the router port {router_port}"
        )


STUB_DOWNLOAD_ARGV = "hf-download-argv.txt"


def _site_for(tmp_path: Path) -> SiteConfig:
    return SiteConfig(
        base_dir=str(tmp_path), engine_env_root=str(tmp_path / "engine-envs")
    )


def _write_hf_stub(site: SiteConfig, profile: ModelProfile, *, confirm: bool) -> Path:
    """Write the ``hf`` the generated script resolves, and return its path.

    ``models info`` answers as *confirm* asks it to; ``download`` records the
    arguments it was handed and writes the config the script validates at the
    end, so a test reads what the script asked the client for without any
    transfer and without a network.
    """
    hf = site.hf_path(profile.engine.type)
    hf.parent.mkdir(parents=True, exist_ok=True)
    hf.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == "models" ]]; then\n'
        '    echo \'{"id": "stub"}\'\n'
        f"    exit {0 if confirm else 1}\n"
        "fi\n"
        'printf "%s\\n" "$@" > "$STUB_DOWNLOAD_ARGV"\n'
        'touch "$STUB_MODEL_DIR/config.json"\n',
        encoding="utf-8",
    )
    hf.chmod(0o755)
    return hf


def _run_download_script(
    script: str, *, model_dir: Path, argv_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "STUB_DOWNLOAD_ARGV": str(argv_path),
            "STUB_MODEL_DIR": str(model_dir),
        },
    )


def test_download_refuses_a_revision_the_repository_does_not_publish(
    tmp_path: Path,
) -> None:
    """A pin the repository cannot resolve must stop the job before transfer.

    The stub's argument file is the instrument: the script exits non-zero and
    that file is absent, so the client was never asked to fetch anything. A
    refusal that arrives after the transfer has begun would leave the file
    behind and read as a failed download rather than as a refused pin.
    """
    profile = load_profile("deepseek-v4-1-flash")
    site = _site_for(tmp_path)
    _write_hf_stub(site, profile, confirm=False)
    argv_path = tmp_path / STUB_DOWNLOAD_ARGV

    result = _run_download_script(
        generate_download_script(profile, site),
        model_dir=site.model_dir(profile),
        argv_path=argv_path,
    )

    assert result.returncode != 0
    assert "refusing to download" in result.stderr
    assert not argv_path.exists(), "the transfer started despite the refusal"


def test_download_script_passes_the_pinned_revision_to_the_client(
    tmp_path: Path,
) -> None:
    """The release the profile names is the release the client is asked for."""
    profile = load_profile("deepseek-v4-1-flash")
    site = _site_for(tmp_path)
    _write_hf_stub(site, profile, confirm=True)
    argv_path = tmp_path / STUB_DOWNLOAD_ARGV

    result = _run_download_script(
        generate_download_script(profile, site),
        model_dir=site.model_dir(profile),
        argv_path=argv_path,
    )

    assert result.returncode == 0, result.stderr
    argv = argv_path.read_text(encoding="utf-8").split()
    assert argv[0] == "download"
    assert argv[argv.index("--revision") + 1] == CHECKPOINT_SHA
    assert argv[argv.index("--local-dir") + 1] == str(site.model_dir(profile))
    # And the check that admits the transfer precedes it in the script.
    script = generate_download_script(profile, site)
    assert script.index("models info") < script.index("hf download")


def test_download_script_pins_nothing_when_no_revision_is_declared() -> None:
    """A profile without a pin keeps the download it generated before.

    The pin is a property of a profile that declares one, so clearing it must
    remove the flag and the check together rather than leave a check resolving
    an empty revision.
    """
    unpinned = load_profile("deepseek-v4-1-flash").model_copy(deep=True)
    unpinned.model.revision = None

    script = generate_download_script(unpinned, SiteConfig())

    assert "--revision" not in script
    assert "REVISION=" not in script
    assert "models info" not in script


def test_serve_dry_run_names_the_weights_directory_and_the_pinned_revision() -> None:
    """The serve dry-run names both halves of the release it serves.

    The weights directory alone does not identify the checkpoint a run loaded,
    so the launcher records the pinned commit beside it and prints it into the
    job log, which is where a reader can tell two deployments apart.
    """
    result = CliRunner().invoke(
        main, ["agent", "serve", "deepseek-v4-1-flash", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert f"MODEL_DIR={CHECKPOINT_MODEL_DIR}" in result.output
    assert CHECKPOINT_SHA in result.output
    assert f"REVISION={CHECKPOINT_SHA}" in result.output
    assert 'echo "  revision: $REVISION"' in result.output


def test_serve_script_records_no_revision_when_the_profile_pins_none() -> None:
    """A profile that pins nothing gains no revision record.

    The record is a property of a profile that declares a pin, so clearing the
    pin must remove it rather than leave the script naming a commit nothing
    fetched.
    """
    unpinned = load_profile("deepseek-v4-1-flash").model_copy(deep=True)
    unpinned.model.revision = None

    script = generate_serve_script(unpinned, SiteConfig(), port=18810)

    assert "REVISION" not in script
    assert "revision" not in script.lower()
    assert f"MODEL_DIR={CHECKPOINT_MODEL_DIR}" in script
    assert "models info" not in script
