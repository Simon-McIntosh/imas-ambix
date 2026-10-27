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
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from imas_ambix.agent import watch_style
from imas_ambix.agent.gpu_watch import view

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


def test_one_shot_takes_one_window_for_the_servers_on_this_host(two_servers):
    probe = _Probe(_sample())
    readings = view.sample_readings(
        two_servers, probe, window_s=2.0, interval_s=1.0, sleep=lambda _s: None
    )
    assert set(readings) == {"gpu-0003"}
    assert probe.calls == [{"interval_s": 1.0, "count": 3}]
    assert len(readings["gpu-0003"].samples) == 3


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
