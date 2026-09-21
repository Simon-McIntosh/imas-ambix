"""The node's card, host and job-table readings, and what an absent probe does.

Coverage
--------
1.  A recorded ``nvidia-smi`` body yields every card quantity the section
    names, from one query.
2.  Cards are labelled by the physical index the process was allocated rather
    than by the order ``nvidia-smi`` read them, asserted on a fixture where the
    two differ and against the environment a batch-step serve actually has --
    the allocation in ``SLURM_JOB_GPUS`` with no ``SLURM_STEP_GPUS`` set.
3.  A quantity a card does not expose is absent from that card rather than
    ``null`` or zero, which is the same rule the engine section follows.
4.  A recorded ``/proc/stat`` pair yields the CPU busy fraction over the
    interval; the first sample of a run has no interval and reports no
    fraction.
5.  A recorded ``/proc/meminfo`` yields the host memory fields in MiB.
6.  A recorded ``squeue`` body yields one row per resident job.
7.  A probe whose command fails, is absent, or returns nothing contributes NO
    section at all -- asserted per source -- because an empty or zeroed section
    reads as a measurement instead of as an absence.
8.  The job table is read on its own slower cadence and omitted from the ticks
    in between, since a repeated reading dates a stale table as current.

The fixtures are recorded bodies.  ``_PROC_STAT``, ``_PROC_MEMINFO`` and
``_SQUEUE`` were captured on this workstation, the job table from the serving
node ``98dci4-gpu-0003`` (three resident jobs: the serve, its router, and a
third job).  ``_CARDS_TWO`` carries the field set and value shapes of a
recorded ``nvidia-smi --query-gpu`` capture, widened to the two-card allocation
the index mapping needs to be visible.
"""

from __future__ import annotations

import pytest

from imas_ambix.agent import node_probe

_CARDS_TWO = (
    "0, 0, 41, 12.62, 72.00, 210, 405, 726, 23034\n"
    "1, 96, 63, 431.55, 700.00, 1980, 2619, 138452, 143771\n"
)

#: The same shape where a card exposes neither a power cap nor a memory clock
#: -- ``nvidia-smi`` prints both as bracketed non-values.
_CARDS_PARTIAL = "0, 96, 63, 431.55, [N/A], 1980, [Not Supported], 138452, 143771\n"

_PROC_STAT = (
    "cpu  421386254 154911551 255295798 17248277704 32872591 14077891 11015351 0 0 0\n"
    "cpu0 11715061 7506393 5333306 350540839 768710 393261 1147375 0 0 0\n"
)

#: The same host five seconds later: user +400, system +100, idle +500,
#: iowait +20. Total +1020 and idle+iowait +520, so busy is 500/1020.
_PROC_STAT_LATER = (
    "cpu  421386654 154911551 255295898 17248278204 32872611 14077891 11015351 0 0 0\n"
)

_PROC_MEMINFO = (
    "MemTotal:       527586420 kB\n"
    "MemFree:        414869608 kB\n"
    "MemAvailable:   441238432 kB\n"
    "Buffers:           12868 kB\n"
    "SwapTotal:      33554432 kB\n"
    "SwapFree:       33554432 kB\n"
)

_SQUEUE = (
    "1271340|murawan|wemm_embed_4b|RUNNING|betelgeuse|7|100G|N/A|5-18:52:35\n"
    "1273253|mcintos|deepseek-v4-1-flash|RUNNING|betelgeuse|12|600G|gres/gpu:4|3-09:46:04\n"
    "1272339|mcintos|ambix-router|RUNNING|betelgeuse|2|8G|N/A|4-09:07:40\n"
)


def _runner(responses: dict[str, str | None]):
    """A probe runner answering by program name, shaped like the real one.

    A ``None`` entry is a command that did not answer, which is what the real
    runner reports for a non-zero exit, an absent binary and a timeout alike.
    """

    def run(argv):
        return responses.get(argv[0])

    return run


def _node_runner(*, cards: str | None = _CARDS_TWO, jobs: str | None = _SQUEUE):
    """A runner for the card and job sources."""
    return _runner({"nvidia-smi": cards, "squeue": jobs})


def _host_runner(stat: str | None, meminfo: str | None):
    """A runner whose two ``cat`` reads answer differently by their argument."""

    def run(argv):
        if argv[0] != "cat":
            return None
        if argv[1] == "/proc/stat":
            return stat
        if argv[1] == "/proc/meminfo":
            return meminfo
        return None

    return run


# ── Cards ────────────────────────────────────────────────────────────


def test_a_recorded_card_body_yields_every_field_the_section_names():
    section = node_probe.read_cards(_node_runner(), env={})

    assert section["count"] == 2
    assert section["cards"] == [
        {
            "index": 0,
            "read_index": 0,
            "utilisation_percent": 0.0,
            "temperature_c": 41.0,
            "power_draw_w": 12.62,
            "power_cap_w": 72.0,
            "sm_clock_mhz": 210.0,
            "mem_clock_mhz": 405.0,
            "memory_used_mib": 726.0,
            "memory_total_mib": 23034.0,
        },
        {
            "index": 1,
            "read_index": 1,
            "utilisation_percent": 96.0,
            "temperature_c": 63.0,
            "power_draw_w": 431.55,
            "power_cap_w": 700.0,
            "sm_clock_mhz": 1980.0,
            "mem_clock_mhz": 2619.0,
            "memory_used_mib": 138452.0,
            "memory_total_mib": 143771.0,
        },
    ]


def test_card_indices_come_from_the_batch_allocation_not_the_read_order():
    """The batch step holds cards 6 and 7; ``nvidia-smi`` numbers them 0 and 1.

    The environment is the one the serve's own launch creates: submitted with
    ``sbatch`` and running the engine inline, so the allocation is in
    ``SLURM_JOB_GPUS`` and there is no ``SLURM_STEP_GPUS`` at all.
    """
    section = node_probe.read_cards(_node_runner(), env={"SLURM_JOB_GPUS": "6,7"})

    assert [card["index"] for card in section["cards"]] == [6, 7]
    assert [card["read_index"] for card in section["cards"]] == [0, 1]
    assert section["index_source"] == "SLURM_JOB_GPUS"
    assert section["allocation"] == {"variable": "SLURM_JOB_GPUS", "value": "6,7"}


def test_a_step_allocation_is_narrower_than_the_job_it_runs_in():
    """A serve under ``srun`` inside a job holds the step's cards, not the job's."""
    section = node_probe.read_cards(
        _node_runner(), env={"SLURM_STEP_GPUS": "6,7", "SLURM_JOB_GPUS": "0,1,2,3,6,7"}
    )

    assert [card["index"] for card in section["cards"]] == [6, 7]
    assert section["index_source"] == "SLURM_STEP_GPUS"


def test_the_visible_device_view_is_never_used_as_the_physical_numbering():
    """``CUDA_VISIBLE_DEVICES`` is remapped to ``0..N-1``, so it labels nothing.

    A card read under it would carry the position in the visible set under the
    name of the silicon, which is the confusion the map exists to remove.
    """
    section = node_probe.read_cards(_node_runner(), env={"CUDA_VISIBLE_DEVICES": "0,1"})

    assert [card["index"] for card in section["cards"]] == [0, 1]
    assert section["index_source"] == "nvidia-smi"
    assert "allocation" not in section


def test_an_allocation_stated_in_another_order_still_pairs_ascending():
    """The pairing is positional against ``nvidia-smi``'s ascending read order."""
    section = node_probe.read_cards(_node_runner(), env={"SLURM_JOB_GPUS": "7,6"})

    assert [card["index"] for card in section["cards"]] == [6, 7]
    assert section["allocation"] == {"variable": "SLURM_JOB_GPUS", "value": "7,6"}


def test_an_unstated_allocation_leaves_the_read_numbering_and_says_so():
    section = node_probe.read_cards(_node_runner(), env={})

    assert [card["index"] for card in section["cards"]] == [0, 1]
    assert section["index_source"] == "nvidia-smi"
    assert "allocation" not in section


def test_an_allocation_that_does_not_describe_the_cards_is_not_guessed():
    """One index for two cards maps nothing, so no card is mislabelled."""
    section = node_probe.read_cards(_node_runner(), env={"SLURM_JOB_GPUS": "6"})

    assert [card["index"] for card in section["cards"]] == [0, 1]
    assert section["index_source"] == "nvidia-smi"
    assert section["allocation"] == {"variable": "SLURM_JOB_GPUS", "value": "6"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0,1,2,3", [0, 1, 2, 3]),
        ("6,7", [6, 7]),
        ("0-3", [0, 1, 2, 3]),
        ("30-31,40", [30, 31, 40]),
        ("", None),
        (None, None),
        ("not indices", None),
        ("7-3", None),
    ],
)
def test_a_slurm_index_list_parses_lists_and_ranges(raw, expected):
    assert node_probe.parse_gpu_index_list(raw) == expected


def test_a_card_quantity_the_device_does_not_expose_is_absent():
    section = node_probe.read_cards(_runner({"nvidia-smi": _CARDS_PARTIAL}), env={})
    card = section["cards"][0]

    assert "power_cap_w" not in card
    assert "mem_clock_mhz" not in card
    assert card["power_draw_w"] == 431.55
    assert card["temperature_c"] == 63.0


# ── Host ─────────────────────────────────────────────────────────────


def test_a_recorded_host_body_yields_cpu_and_memory_fields():
    first = node_probe.read_host(_host_runner(_PROC_STAT, _PROC_MEMINFO))
    earlier = node_probe.parse_proc_stat(_PROC_STAT)
    later = node_probe.parse_proc_stat(_PROC_STAT_LATER)
    second = node_probe.build_host_section(later, {}, earlier)

    assert earlier is not None and later is not None
    assert first["cpu_busy_jiffies"] == 856686845.0
    assert first["cpu_total_jiffies"] == 18137837140.0
    assert "cpu_busy_fraction" not in first
    assert second["cpu_busy_fraction"] == 0.490196
    assert first["memory_total_mib"] == 515221.113
    assert first["memory_available_mib"] == 430896.906
    assert first["memory_free_mib"] == 405146.102
    assert first["memory_used_mib"] == 84324.207
    assert first["memory_used_percent"] == 16.3666
    assert first["swap_total_mib"] == 32768.0


def test_an_unchanged_cpu_sample_reports_no_fraction():
    """A zero-length interval carries no rate, refused rather than zeroed."""
    times = node_probe.parse_proc_stat(_PROC_STAT)

    assert node_probe.cpu_busy_fraction(times, times) is None
    assert node_probe.cpu_busy_fraction(None, times) is None


def test_a_meminfo_without_an_available_column_omits_the_derived_fields():
    section = node_probe.build_host_section(
        None, node_probe.parse_meminfo("MemTotal:  1024 kB\n")
    )

    assert section == {"memory_total_mib": 1.0}


# ── Job table ────────────────────────────────────────────────────────


def test_a_recorded_job_table_yields_one_row_per_job():
    section = node_probe.read_jobs("98dci4-gpu-0003", _node_runner())

    assert section["hostname"] == "98dci4-gpu-0003"
    assert section["count"] == 3
    assert section["jobs"][1] == {
        "job_id": "1273253",
        "user": "mcintos",
        "name": "deepseek-v4-1-flash",
        "state": "RUNNING",
        "partition": "betelgeuse",
        "cpus": 12,
        "memory": "600G",
        "gres": "gres/gpu:4",
        "elapsed": "3-09:46:04",
    }
    assert section["jobs"][0]["gres"] == "N/A"


def test_a_node_holding_no_jobs_is_a_reading_and_is_kept():
    section = node_probe.read_jobs("98dci4-gpu-0003", _node_runner(jobs=""))

    assert section == {"hostname": "98dci4-gpu-0003", "count": 0, "jobs": []}


def test_the_job_table_carries_the_node_it_describes():
    calls: list[list[str]] = []

    def run(argv):
        calls.append(list(argv))
        return _SQUEUE

    node_probe.read_jobs("98dci4-gpu-0003", run)

    assert calls == [
        ["squeue", "-h", "-w", "98dci4-gpu-0003", "-o", node_probe.SQUEUE_FORMAT]
    ]


# ── A source that did not answer ─────────────────────────────────────


def test_a_card_query_that_did_not_answer_contributes_no_card_section():
    probe = node_probe.NodeProbe(run=_node_runner(cards=None), env={}, hostname="n")

    sections = probe.sample(0.0)

    assert "cards" not in sections
    assert set(sections) == {"jobs"}


def test_a_failed_host_read_contributes_no_host_section():
    probe = node_probe.NodeProbe(run=_host_runner(None, None), env={}, hostname="n")

    assert "host" not in probe.sample(0.0)


def test_a_job_query_that_did_not_answer_contributes_no_job_section():
    probe = node_probe.NodeProbe(run=_node_runner(jobs=None), env={}, hostname="n")

    assert "jobs" not in probe.sample(0.0)


@pytest.mark.parametrize(
    ("stat", "meminfo", "present", "absent"),
    [
        (
            _PROC_STAT,
            None,
            {"cpu_busy_jiffies", "cpu_total_jiffies"},
            {"memory_total_mib"},
        ),
        (
            None,
            _PROC_MEMINFO,
            {"memory_total_mib", "memory_used_percent"},
            {"cpu_total_jiffies"},
        ),
    ],
)
def test_a_host_with_one_unreadable_source_keeps_only_what_it_read(
    stat, meminfo, present, absent
):
    section = node_probe.read_host(_host_runner(stat, meminfo))

    assert present <= set(section)
    assert not (absent & set(section))


def test_a_probe_with_nothing_at_all_to_report_emits_no_sections():
    """The whole-node case: no source answered, so the row carries no node keys."""
    probe = node_probe.NodeProbe(
        run=_runner({"nvidia-smi": None, "cat": None, "squeue": None}),
        env={},
        hostname="n",
    )

    assert probe.sample(0.0) == {}


def test_a_nonzero_exit_is_not_a_reading_even_when_the_command_printed():
    """The real runner, which is the guard every probe above rests on."""
    assert node_probe.run_capture(["true"]) == ""
    assert node_probe.run_capture(["bash", "-c", "echo hi; exit 3"]) is None
    assert node_probe.run_capture(["ambix-no-such-binary-probe"]) is None


# ── Cadence ──────────────────────────────────────────────────────────


def test_the_job_table_is_read_on_its_own_slower_cadence():
    calls: list[str] = []

    def run(argv):
        calls.append(argv[0])
        return {
            "nvidia-smi": _CARDS_TWO,
            "cat": _PROC_STAT,
            "squeue": _SQUEUE,
        }.get(argv[0])

    probe = node_probe.NodeProbe(run=run, env={}, hostname="n", job_interval_s=60.0)
    first = probe.sample(0.0)
    between = probe.sample(5.0)
    after = probe.sample(60.0)

    assert set(first) == {"cards", "host", "jobs"}
    assert set(between) == {"cards", "host"}
    assert set(after) == {"cards", "host", "jobs"}
    assert calls.count("squeue") == 2
