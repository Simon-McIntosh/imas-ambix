"""gpu-watch renders every declared server, and says when there is nothing there.

Coverage
--------
1.  A two-server configuration renders two stacked blocks, each carrying its
    own server's header, card table, host row and engine rows.
2.  ``--server NAME`` limits the view to that one server's block.
3.  Reservation-group labels come from the server's declaration, so a document
    declaring ``grpA`` shows ``grpA`` and one declaring nothing shows a dash
    rather than a constant baked into the renderer.
4.  A declared server with no running job prints ``no job to sample`` instead
    of an empty card table, because an empty table reads as a node with no
    cards rather than as a server nobody is serving on.
5.  A server this host is not is not drawn from the local node's readings: the
    block says it was not sampled here.
6.  Card figures pair samples by physical index and report the window mean
    beside the latest reading; a quantity no sample carried is a dash.
7.  A one-shot sample takes one window for every server declared on this host
    and none for a server on another host.
8.  The package carries no serve-panel symbols (``serve_grid``,
    ``callers_table``, ``serve_ledger_table``) and no hard-coded node name,
    asserted by reading the package's own sources.
9.  The style module is shared with llm-watch: both views resolve their
    percentage formatter to the same function object.
10. A step that writes more than a pipe buffer to stderr still streams its
    samples, so a warning from ``srun`` cannot freeze the view; a step that
    stops streaming reports its last stderr line as the reason.
11. Two declarations naming one machine in different spellings are sampled by
    one step and both render its cards, because the live loop keys the pool by
    the machine name for ``ensure``, ``drain`` and ``state`` alike.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import pytest

from imas_ambix.agent import watch_style
from imas_ambix.agent.gpu_watch import launcher, view

PACKAGE = Path(view.__file__).parent

ENDPOINTS = [
    {
        "host": "98dci4-gpu-0003.iter.org",
        "port": 18810,
        "model_id": "deepseek-v4.1-flash",
        "accelerator_count": 4,
        "accelerator_family": "H200",
    },
    {
        "host": "98dci4-gpu-0002.iter.org",
        "port": 18765,
        "model_id": "qwen3-embedding",
        "accelerator_count": 1,
        "accelerator_family": "H100",
    },
]


def _server(name, *, engines, groups=(), power_cap=None):
    return {
        "name": name,
        "host": f"98dci4-{name}.iter.org",
        "accelerator_family": "H200",
        "accelerator_count": 8,
        "power_cap_watts": power_cap,
        "reservation_groups": list(groups),
        "engines": list(engines),
    }


def _document(servers):
    return {"servers": servers, "endpoints": ENDPOINTS}


def _sample(*, with_cards=True, jobs=None):
    sample = {
        "host": {
            "cpu_busy_fraction": 0.25,
            "memory_used_mib": 204800.0,
            "memory_used_percent": 13.3,
        },
        "jobs": jobs
        if jobs is not None
        else {
            "hostname": "98dci4-gpu-0003.iter.org",
            "count": 1,
            "jobs": [
                {
                    "job_id": "1291588",
                    "user": "mcintos",
                    "name": "ambix-serve",
                    "state": "RUNNING",
                    "cpus": 8,
                    "gres": "gpu:4",
                    "elapsed": "02:11:04",
                }
            ],
        },
    }
    if with_cards:
        sample["cards"] = {
            "count": 2,
            "cards": [
                {
                    "index": 0,
                    "name": "NVIDIA H200",
                    "utilisation_percent": 40.0,
                    "temperature_c": 61.0,
                    "power_draw_w": 420.0,
                    "power_cap_w": 600.0,
                    "memory_used_mib": 90000.0,
                    "memory_total_mib": 143771.0,
                },
                {
                    "index": 1,
                    "name": "NVIDIA H200",
                    "utilisation_percent": 20.0,
                    "temperature_c": 55.0,
                    "power_draw_w": 300.0,
                    "power_cap_w": 600.0,
                    "memory_used_mib": 60000.0,
                    "memory_total_mib": 143771.0,
                },
            ],
        }
    return sample


def _no_job_sample():
    return _sample(
        with_cards=False,
        jobs={"hostname": "98dci4-gpu-0002.iter.org", "count": 0, "jobs": []},
    )


def _readings():
    """The readings fixture the render tests are pointed at.

    Server ``gpu-0003`` is on this host and was sampled with cards; server
    ``gpu-0002`` is a server with no running job. Pointing this fixture at the
    no-job reading is the mutation the negative control applies: the card-table
    assertion then fails while the ``no job to sample`` assertion still passes.
    """
    return {
        "gpu-0003": view.Reading((_sample(),)),
        "gpu-0002": view.Reading((_no_job_sample(),)),
    }


@pytest.fixture()
def two_servers():
    return _document(
        [
            _server(
                "gpu-0003",
                engines=["gpu-0003.iter.org:18810"],
                groups=["gpu_0003_grpA"],
                power_cap=5600,
            ),
            _server("gpu-0002", engines=["gpu-0002.iter.org:18765"]),
        ]
    )


def test_two_server_configuration_renders_two_stacked_blocks(two_servers):
    text = view.render_text(two_servers, _readings())
    assert "gpu-0003" in text and "gpu-0002" in text
    assert text.count("conditions") == 1, text
    assert text.count("util now") == 1, text
    assert text.index("gpu-0003") < text.index("gpu-0002")
    assert "deepseek-v4.1-flash" in text
    assert "qwen3-embedding" in text


def test_server_option_limits_the_view_to_one(two_servers):
    text = view.render_text(two_servers, _readings(), server="gpu-0002")
    assert "gpu-0002" in text
    assert "gpu-0003" not in text
    assert "qwen3-embedding" in text


def test_reservation_group_labels_come_from_the_declaration(two_servers):
    text = view.render_text(two_servers, _readings())
    header = text.splitlines()[0]
    assert "gpu_0003_grpA" in header
    other = [line for line in text.splitlines() if line.startswith("gpu-0002")]
    assert other and "groups" in other[0] and "—" in other[0]


def test_a_server_with_no_running_job_says_so_instead_of_an_empty_card_table(
    two_servers,
):
    text = view.render_text(two_servers, _readings(), server="gpu-0002")
    assert view.NO_JOB in text
    assert "util now" not in text
    assert "temperature" not in text


def test_a_server_on_another_host_is_not_drawn_from_local_readings():
    document = _document([_server("gpu-0009", engines=[])])
    text = view.render_text(document, {"gpu-0003": view.Reading((_sample(),))})
    assert view.NO_READING in text
    assert "util now" not in text


def test_card_figures_carry_the_window_mean_beside_the_latest_reading():
    first = _sample()
    second = _sample()
    second["cards"]["cards"][0]["utilisation_percent"] = 60.0
    second["cards"]["cards"][1]["utilisation_percent"] = 20.0
    reading = view.Reading((first, second))
    rows = view.card_figures(reading)
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[0]["util_now"] == 60.0
    assert rows[0]["util_mean"] == 50.0
    assert rows[1]["util_now"] == 20.0
    assert rows[1]["util_mean"] == 20.0


def test_a_declared_power_cap_is_a_watt_reading_scaled_to_kilowatts():
    """The declaration is watts and the draw is kilowatts, so both must be one unit.

    A 700 W declaration is a 0.70 kW cap. Read unscaled it renders 700 kW and a
    share a thousand times too small, which is what a declared cap used to do.
    """
    sample = _sample()
    cards = sample["cards"]["cards"]
    cards[0]["power_draw_w"] = 420.0
    cards[1]["power_draw_w"] = 0.0

    figures = view.conditions(view.Reading((sample,)), 700)
    line = view.conditions_line(view.Reading((sample,)), 700).plain

    assert figures["kilowatts"] == 0.42
    assert figures["cap_kilowatts"] == 0.7
    assert figures["cap_percent"] == 60.0
    assert "0.70 kW cap" in line
    assert "60% of cap" in line


class _Probe:
    """A probe stand-in that answers one recorded sample per tick."""

    hostname = "98dci4-gpu-0003.iter.org"

    def __init__(self, sample):
        self.sample = sample
        self.calls = []

    def stream(self, interval_s, *, count=None, clock=None, sleep=None):
        self.calls.append({"interval_s": interval_s, "count": count})
        produced = 0
        while count is None or produced < count:
            yield self.sample
            produced += 1


class _NoRemote:
    """A sampler that finds no job elsewhere, so no step is opened."""

    def read(self, host, *, window_s, interval_s):
        return launcher.StepReading()


def test_one_servers_window_is_taken_in_process_and_the_other_is_not(
    two_servers,
):
    probe = _Probe(_sample())
    readings = view.sample_readings(
        two_servers,
        probe,
        window_s=2.0,
        interval_s=1.0,
        sleep=lambda _s: None,
        remote=_NoRemote(),
    )
    assert set(readings) == {"gpu-0003", "gpu-0002"}
    assert probe.calls == [{"interval_s": 1.0, "count": 3}]
    assert len(readings["gpu-0003"].samples) == 3
    # The server elsewhere holds no job to attach to: it renders as NO_JOB.
    assert view.NO_JOB in view.render_text(two_servers, readings, server="gpu-0002")


def test_the_package_carries_no_serve_panel_symbol_and_no_node_name():
    sources = {path: path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py")}
    assert sources
    for banned in ("serve_grid", "callers_table", "serve_ledger_table"):
        offenders = [str(path) for path, text in sources.items() if banned in text]
        assert not offenders, f"{banned} in {offenders}"
    for path, text in sources.items():
        for spelling in ("98dci4", "gpu-0003", "gpu_0003"):
            assert spelling not in text, f"{spelling} hard-coded in {path}"


def test_the_style_module_is_shared_with_llm_watch():
    llm_watch = importlib.import_module("imas_ambix.agent.llm_watch")
    assert view.pct is watch_style.pct
    assert "watch_style" in Path(llm_watch.__file__).read_text(encoding="utf-8")


# ── The launcher: sampling another server through its own running job ──


def _job(argv, *, timeout_s=15.0):
    """A running job of ours on every server, answered without a scheduler."""
    assert argv[0] == "squeue"
    return "1294597|gres/gpu:4\n"


def _install_fake_srun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    samples_path: Path | None = None,
    term_path: Path | None = None,
) -> Path:
    """Write a fake ``srun`` onto ``PATH`` and return its argv log.

    *mode* selects the step's behaviour: ``samples`` prints the given JSON
    lines and exits, ``noisy`` writes more than a pipe buffer to stderr first
    and then prints the same lines, ``empty`` exits at once without streaming,
    ``noisy-empty`` writes one stderr line and exits without streaming, and
    ``hang`` streams nothing until it is signalled -- recording that it was
    terminated.
    ``noisy`` is what a real ``srun`` does when it warns: the launcher must
    drain stderr or the step blocks on a full pipe and never streams a sample.
    """
    log = tmp_path / "argv.log"
    samples_path = samples_path or tmp_path / "samples.jsonl"
    term_path = term_path or tmp_path / "term.log"
    script = tmp_path / "srun"
    script.write_text(
        "#!/bin/sh\n"
        f"trap 'echo term >> \"{term_path}\"; exit 0' TERM\n"
        f"printf '%s\\n' \"$*\" >> '{log}'\n"
        f"case '{mode}' in\n"
        f"  samples) cat '{samples_path}' ;;\n"
        "  noisy) i=0; while [ $i -lt 3000 ]; do "
        'echo "step warning line $i that is long enough to fill a pipe" >&2; '
        f"i=$((i+1)); done; cat '{samples_path}' ;;\n"
        "  empty) ;;\n"
        "  noisy-empty) echo 'srun: error: could not open the sampler' >&2 ;;\n"
        "  hang) while :; do sleep 0.02; done ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    return log


def _remote_document():
    return _document([_server("gpu-0003", engines=["gpu-0003.iter.org:18810"])])


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_the_step_argv_names_the_job_overlap_and_the_module_entry(
    tmp_path, monkeypatch
):
    samples = tmp_path / "one.jsonl"
    samples.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    log = _install_fake_srun(
        tmp_path, monkeypatch, mode="samples", samples_path=samples
    )
    sampler = launcher.RemoteSampler(run=_job, user="tester")

    step = sampler.read("98dci4-gpu-0003", window_s=1.0, interval_s=1.0)

    argv = log.read_text(encoding="utf-8")
    assert step.job_id == "1294597"
    assert "--jobid" in argv and "1294597" in argv
    assert "--overlap" in argv
    assert "-m" in argv and launcher.SAMPLER_MODULE in argv
    # A step into a full allocation only starts with the CPU-binding flag.
    assert "--cpu-bind=none" in argv


def test_samples_streamed_by_the_step_are_drawn_into_the_card_table(
    tmp_path, monkeypatch
):
    samples = tmp_path / "two.jsonl"
    samples.write_text(
        "\n".join(json.dumps(_sample()) for _ in range(2)) + "\n", encoding="utf-8"
    )
    _install_fake_srun(tmp_path, monkeypatch, mode="samples", samples_path=samples)
    document = _remote_document()
    sampler = launcher.RemoteSampler(run=_job, user="tester")

    readings = view.sample_readings(
        document,
        remote=sampler,
        hostname="98dci4-clu-2068",
        window_s=1.0,
        interval_s=1.0,
    )
    text = view.render_text(document, readings)

    assert "util now" in text, text
    assert "NVIDIA H200" in text, text
    assert "not sampled" not in text, text
    assert view.NO_JOB not in text, text


def test_the_step_is_terminated_and_reaped_when_the_watcher_exits(
    tmp_path, monkeypatch
):
    term = tmp_path / "terminated.log"
    _install_fake_srun(tmp_path, monkeypatch, mode="hang", term_path=term)
    pool = launcher.SamplerPool(run=_job, user="tester", interval_s=0.05)

    pool.ensure("98dci4-gpu-0003")
    # The script records its argv before it starts streaming, so its presence
    # says the step is running and the trap is armed.
    assert _wait_until(lambda: (tmp_path / "argv.log").exists())
    pool.close()

    assert term.exists(), "the step was left running past the watcher's exit"


def test_a_document_whose_step_dies_is_restarted_once(tmp_path, monkeypatch):
    samples = tmp_path / "one.jsonl"
    samples.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    _install_fake_srun(tmp_path, monkeypatch, mode="samples", samples_path=samples)
    pool = launcher.SamplerPool(run=_job, user="tester", interval_s=0.01)
    host = "98dci4-gpu-0003"

    try:
        assert _wait_until(
            lambda: (pool.ensure(host), pool.launches(host) == 2)[1]
        ), "a step dropped after its first launch was not restarted"

        # The budget is one restart: further drops must not spend a third step.
        for _ in range(10):
            pool.ensure(host)
            time.sleep(0.02)
        assert pool.launches(host) == 2
    finally:
        pool.close()


def test_a_step_that_exits_without_streaming_renders_not_sampled_with_a_reason(
    tmp_path, monkeypatch
):
    """A running server whose step answered nothing must say so.

    Not an empty card table (which reads as a node with no cards), and not
    ``no job to sample`` (which is a lie: a job is running and was attached to).
    """
    _install_fake_srun(tmp_path, monkeypatch, mode="empty")
    document = _remote_document()
    sampler = launcher.RemoteSampler(run=_job, user="tester")

    readings = view.sample_readings(
        document,
        remote=sampler,
        hostname="98dci4-clu-2068",
        window_s=1.0,
        interval_s=1.0,
    )
    text = view.render_text(document, readings)

    assert "not sampled" in text, text
    assert launcher.NO_SAMPLES_REASON in text, text
    assert "util now" not in text, text
    assert view.NO_JOB not in text, text


def _two_spellings_document():
    """One machine, declared twice: bare and fully qualified."""
    return {
        "servers": [
            {
                "name": "gpu-0003",
                "host": "98dci4-gpu-0003",
                "accelerator_family": "H200",
                "accelerator_count": 8,
                "engines": [],
            },
            {
                "name": "gpu-0003-fqdn",
                "host": "98dci4-gpu-0003.iter.org",
                "accelerator_family": "H200",
                "accelerator_count": 8,
                "engines": [],
            },
        ],
        "endpoints": [],
    }


def _live_readings(document, pool, *, capacity=4):
    servers = view.servers_of(document)
    local, groups = view._live_groups(servers, "98dci4-clu-2068")
    history: dict = {}

    def tick():
        return view._live_tick(
            servers,
            local=local,
            remote_groups=groups,
            probe=None,
            pool=pool,
            history=history,
            capacity=capacity,
        )

    return servers, groups, tick


def test_a_steps_stderr_larger_than_a_pipe_buffer_does_not_stop_the_stream(
    tmp_path, monkeypatch
):
    """A step that writes to stderr must still be able to stream its samples.

    ``srun`` warns on stderr. With nobody reading that pipe the step blocks as
    soon as it fills, so the cards freeze while the pool still calls the step
    live -- the view shows a server that was working a moment ago and then
    never changes again.
    """
    samples = tmp_path / "noisy.jsonl"
    samples.write_text(
        "\n".join(json.dumps(_sample()) for _ in range(2)) + "\n", encoding="utf-8"
    )
    _install_fake_srun(tmp_path, monkeypatch, mode="noisy", samples_path=samples)
    document = _remote_document()
    pool = launcher.SamplerPool(run=_job, user="tester", interval_s=0.05)

    try:
        _, groups, tick = _live_readings(document, pool)
        assert list(groups) == ["98dci4-gpu-0003"]
        assert _wait_until(
            lambda: bool(tick()["gpu-0003"].samples), timeout_s=15.0
        ), "the step filled its stderr pipe and stopped streaming"
        text = view.render_text(document, tick())
        assert "util now" in text, text
        assert "NVIDIA H200" in text, text
        assert "not sampled" not in text, text
    finally:
        pool.close()


def test_a_step_that_stops_streaming_names_its_last_stderr_line(
    tmp_path, monkeypatch
):
    _install_fake_srun(tmp_path, monkeypatch, mode="noisy-empty")
    document = _remote_document()
    pool = launcher.SamplerPool(run=_job, user="tester", interval_s=0.05)

    try:
        _, _, tick = _live_readings(document, pool)
        assert _wait_until(
            lambda: "could not open the sampler" in tick()["gpu-0003"].unavailable,
            timeout_s=10.0,
        ), "a step that stopped streaming did not report why"
        text = view.render_text(document, tick())
        assert "not sampled: srun: error: could not open the sampler" in text, text
    finally:
        pool.close()


def test_two_spellings_of_one_machine_render_the_same_cards(tmp_path, monkeypatch):
    """One machine declared twice is sampled once and drawn twice.

    The live loop keys its pool by the machine name, so both declarations read
    the step and both show the cards. Keying by each declaration's own spelling
    launches one step and looks up another, and the second block then waits for
    samples that are already arriving.
    """
    samples = tmp_path / "one.jsonl"
    samples.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    _install_fake_srun(tmp_path, monkeypatch, mode="samples", samples_path=samples)
    document = _two_spellings_document()
    pool = launcher.SamplerPool(run=_job, user="tester", interval_s=0.05)

    try:
        _, groups, tick = _live_readings(document, pool)
        assert list(groups) == ["98dci4-gpu-0003"], groups
        assert _wait_until(
            lambda: all(
                reading.samples for reading in tick().values()
            ),
            timeout_s=10.0,
        ), "a declaration of the sampled machine rendered nothing"
        # One target under the machine name; the second spelling never became a
        # key of its own, which is the defect this guards.
        assert pool.launches("98dci4-gpu-0003") >= 1
        assert pool.launches("98dci4-gpu-0003.iter.org") == 0
        text = view.render_text(document, tick())
        assert text.count("NVIDIA H200") >= 2, text
        assert "not sampled" not in text, text
    finally:
        pool.close()
