"""Tests for the CPU throttling sampler over cgroup v2 text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.agent import throttling
from imas_ambix.agent.throttling import (
    CpuMax,
    CpuStat,
    HostIdentity,
    Sample,
    Unmeasured,
    _counters_payload,
    _host_payload,
    _parser,
    format_cpu_max,
    interval_delta,
    interval_throttled_share,
    main,
    parse_boot_id,
    parse_cpu_max,
    parse_cpu_stat,
    parse_uptime,
    permitted_thread_usec,
    read_host,
    read_sample,
    throttled_share,
    throttled_share_over_elapsed,
)

# The account slice on the login node: four cores out of each 100 ms period.
_QUOTA_ED_MAX = "400000 100000"

# A machine identity fixture: one host, one boot of it, and how long that boot
# has run. The readings are attributed to this rather than to whatever machine
# the suite happens to run on, so the fields are asserted over known values.
_HOST = HostIdentity(
    hostname="srv-alpha",
    boot_id="5f2c9d84-1e3a-4b70-9c11-8a3f2d6b04c7",
    uptime_seconds=3_534_804.72,
)

_UPTIME_TEXT = "3534804.72 987654.32\n"

# The kernel adds fields this module does not report on; a host on a later
# kernel must still produce a reading.
_CPU_STAT_TEXT = """\
usage_usec 4846187612533
user_usec 3238892601672
system_usec 1607295010861
core_sched.force_idle_usec 0
nr_periods 1000
nr_throttled 250
throttled_usec 100000000
nr_bursts 0
burst_usec 0
"""

# Captured from the held allocation's control group
# /sys/fs/cgroup/user.slice/user-39486.slice on 98dci4-clu-2002: no cpu.max,
# and a cpu.stat that states no period counter at all. This is one side of the
# comparison the sampler exists to take.
_CPU_STAT_WITHOUT_PERIODS = """\
usage_usec 401401
user_usec 244704
system_usec 156696
core_sched.force_idle_usec 0
"""

# Captured from the root control group /sys/fs/cgroup on the login node
# 98dci4-srv-1006: no cpu.max of its own, and a cpu.stat carrying the three
# period counters with genuinely zero values. The root group accounts the whole
# machine's run, so it keeps those counters while declaring no ceiling that
# could be exceeded.
_CPU_STAT_ROOT_WITHOUT_CEILING = """\
usage_usec 8962276463874
user_usec 6049293696034
system_usec 2912982767840
core_sched.force_idle_usec 4197
nr_periods 0
nr_throttled 0
throttled_usec 0
nr_bursts 0
burst_usec 0
"""


def _stat(
    *,
    usage_usec: int = 1_000_000,
    nr_periods: int = 1000,
    nr_throttled: int = 250,
    throttled_usec: int = 100_000_000,
) -> CpuStat:
    return CpuStat(
        usage_usec=usage_usec,
        nr_periods=nr_periods,
        nr_throttled=nr_throttled,
        throttled_usec=throttled_usec,
    )


class TestParseCpuMax:
    def test_quota_ed_ceiling_states_quota_and_period(self):
        cpu_max = parse_cpu_max(_QUOTA_ED_MAX)

        assert cpu_max.quota_usec == 400_000
        assert cpu_max.period_usec == 100_000
        assert cpu_max.can_throttle is True

    def test_literal_max_means_no_ceiling_rather_than_a_large_one(self):
        cpu_max = parse_cpu_max("max 100000")

        assert cpu_max.quota_usec is None
        assert cpu_max.period_usec == 100_000
        assert cpu_max.can_throttle is False

    def test_format_round_trips_both_forms(self):
        for text in (_QUOTA_ED_MAX, "max 100000"):
            assert format_cpu_max(parse_cpu_max(text)) == text

    @pytest.mark.parametrize(
        "text",
        ["400000", "400000 100000 7", "", "400000 abc", "400000 0", "400000 -1"],
    )
    def test_malformed_ceiling_is_refused(self, text):
        with pytest.raises(ValueError):
            parse_cpu_max(text)


class TestParseCpuStat:
    def test_reads_the_reported_counters_and_ignores_the_rest(self):
        stat = parse_cpu_stat(_CPU_STAT_TEXT)

        assert stat.usage_usec == 4_846_187_612_533
        assert stat.nr_periods == 1000
        assert stat.nr_throttled == 250
        assert stat.throttled_usec == 100_000_000

    def test_a_missing_counter_is_refused_rather_than_defaulted_to_zero(self):
        text = _CPU_STAT_TEXT.replace("nr_throttled 250\n", "")

        with pytest.raises(ValueError, match="nr_throttled"):
            parse_cpu_stat(text)


class TestParseUptime:
    def test_reads_the_first_field_and_ignores_the_idle_total(self):
        # The second field sums idle time over every processor, which is not
        # how long the machine has been running.
        assert parse_uptime(_UPTIME_TEXT) == pytest.approx(3_534_804.72)

    def test_a_machine_that_has_just_booted_states_a_small_age(self):
        assert parse_uptime("12.34 100.00\n") == pytest.approx(12.34)

    def test_surrounding_whitespace_does_not_change_the_age(self):
        assert parse_uptime("  42.5\n") == pytest.approx(42.5)

    @pytest.mark.parametrize("text", ["", "   ", "\n", "not-a-number 5.0"])
    def test_text_that_states_no_age_is_refused(self, text):
        with pytest.raises(ValueError):
            parse_uptime(text)


class TestParseBootId:
    def test_takes_the_identifier_apart_from_whitespace(self):
        assert parse_boot_id(f"{_HOST.boot_id}\n") == _HOST.boot_id

    @pytest.mark.parametrize("text", ["", "\n", "   "])
    def test_blank_text_is_refused_rather_than_reported_as_a_blank_boot(self, text):
        with pytest.raises(ValueError):
            parse_boot_id(text)


class TestHostPayload:
    def test_carries_the_machine_the_boot_and_the_age(self):
        assert _host_payload(_HOST) == {
            "hostname": "srv-alpha",
            "boot_id": _HOST.boot_id,
            "uptime_seconds": pytest.approx(3_534_804.72),
        }

    def test_a_reboot_between_two_readings_is_visible_in_the_block(self):
        # Two readings of cumulative counters can be compared only if a reboot
        # between them is visible: those counters restart at boot, so a share
        # that fell may describe a new boot rather than a recovered machine.
        after_reboot = HostIdentity(
            hostname=_HOST.hostname,
            boot_id="0b6d51a9-7c48-42e0-b3f5-9d1cbe27a804",
            uptime_seconds=120.0,
        )

        assert _host_payload(after_reboot)["boot_id"] != _host_payload(_HOST)["boot_id"]
        assert (
            _host_payload(after_reboot)["uptime_seconds"]
            < _host_payload(_HOST)["uptime_seconds"]
        )


class TestThrottledShare:
    def test_cumulative_share_is_stalled_thread_time_over_permitted_thread_time(self):
        # Four cores permitted over 1000 periods is 4e8 usec of thread-time;
        # 1e8 of it was refused.
        stat = _stat(nr_periods=1000, throttled_usec=100_000_000)

        share = throttled_share(stat, parse_cpu_max(_QUOTA_ED_MAX))

        assert share == pytest.approx(0.25)

    def test_permitted_thread_time_is_the_quota_times_the_periods(self):
        # The period itself cancels against the quota, so the answer is the
        # quota multiplied by the count of accounted periods.
        cpu_max = parse_cpu_max("400000 100000")

        assert permitted_thread_usec(_stat(nr_periods=1000), cpu_max) == 400_000_000

    def test_unquota_ed_group_has_an_undefined_share_not_a_zero_share(self):
        # With no ceiling nothing can be refused, and a measured zero would
        # assert that the group was offered a ceiling and never hit it.
        stat = _stat(nr_periods=1000, throttled_usec=0)

        share = throttled_share(stat, parse_cpu_max("max 100000"))

        assert share is Unmeasured.UNBOUNDED

    def test_share_passes_one_when_more_tasks_stall_than_the_ceiling_admits(self):
        # 1000 periods permit 4e8 usec of thread-time; 5e8 was stopped, because
        # more tasks were runnable than the quota admitted and they stalled
        # together for the rest of each period.
        stat = _stat(nr_periods=1000, throttled_usec=500_000_000)

        share = throttled_share(stat, parse_cpu_max(_QUOTA_ED_MAX))

        assert share == pytest.approx(1.25)

    def test_no_accounted_period_yields_an_undefined_share(self):
        # A freshly created group has no periods to divide by; that is an
        # absence of a window, not a window in which nothing was refused.
        stat = _stat(nr_periods=0, throttled_usec=0)

        share = throttled_share(stat, parse_cpu_max(_QUOTA_ED_MAX))

        assert share is Unmeasured.NO_PERIODS

    def test_the_two_absences_are_distinguishable_and_neither_is_a_number(self):
        # Both cases have no share to state, and they call for different
        # responses: an unbounded group will never produce one, while an
        # unaccounted one may at the next reading. A single missing value would
        # force a caller to guess which it is looking at.
        unquota_ed = _stat(nr_periods=1000)
        unaccounted = _stat(nr_periods=0)

        unbounded = throttled_share(unquota_ed, parse_cpu_max("max 100000"))
        no_periods = throttled_share(unaccounted, parse_cpu_max(_QUOTA_ED_MAX))

        assert unbounded is not no_periods
        assert not isinstance(unbounded, float)
        assert not isinstance(no_periods, float)

    def test_an_unbounded_group_with_no_periods_reports_the_permanent_cause(self):
        # Both absences hold here; the ceiling's absence is the one no later
        # reading can remove, so it is the one reported.
        stat = _stat(nr_periods=0)

        share = throttled_share(stat, parse_cpu_max("max 100000"))

        assert share is Unmeasured.UNBOUNDED


class TestIntervalShare:
    def _samples(self) -> tuple[Sample, Sample]:
        cpu_max = parse_cpu_max(_QUOTA_ED_MAX)
        first = Sample(
            cpu_max=cpu_max,
            stat=_stat(nr_periods=1000, throttled_usec=100_000_000),
        )
        # 200 further periods permitted 8e7 usec of thread-time, of which
        # 6e7 was refused.
        second = Sample(
            cpu_max=cpu_max,
            stat=_stat(nr_periods=1200, throttled_usec=160_000_000),
        )
        return first, second

    def test_delta_counters_are_the_difference_between_two_samples(self):
        first, second = self._samples()

        delta = interval_delta(first, second)

        assert delta.nr_periods == 200
        assert delta.throttled_usec == 60_000_000

    def test_interval_share_uses_the_delta_not_the_cumulative_counters(self):
        first, second = self._samples()

        assert interval_throttled_share(first, second) == pytest.approx(0.75)

    def test_a_counter_going_backwards_is_refused(self):
        # A recreated control group restarts its counters; subtracting those
        # would report a negative interval rather than a new one.
        first, second = self._samples()

        with pytest.raises(ValueError, match="nr_periods"):
            interval_delta(second, first)

    def test_an_unquota_ed_interval_has_an_undefined_share(self):
        cpu_max = parse_cpu_max("max 100000")
        first = Sample(cpu_max=cpu_max, stat=_stat())
        second = Sample(cpu_max=cpu_max, stat=_stat(nr_periods=1200))

        share = interval_throttled_share(first, second)

        assert share is Unmeasured.UNBOUNDED

    def test_an_interval_with_no_elapsed_period_has_an_undefined_share(self):
        # Two reads of a group that was already idle in both: nothing accrued
        # between them, so the span holds no window.
        cpu_max = parse_cpu_max(_QUOTA_ED_MAX)
        first = Sample(cpu_max=cpu_max, stat=_stat(nr_periods=1000))
        second = Sample(cpu_max=cpu_max, stat=_stat(nr_periods=1000))

        share = interval_throttled_share(first, second)

        assert share is Unmeasured.NO_PERIODS


class TestShareOverElapsedSpan:
    """The span-length form: stall rise over the ceiling's capacity for it."""

    def _samples(self) -> tuple[Sample, Sample]:
        cpu_max = parse_cpu_max(_QUOTA_ED_MAX)
        first = Sample(cpu_max=cpu_max, stat=_stat(throttled_usec=100_000_000))
        second = Sample(cpu_max=cpu_max, stat=_stat(throttled_usec=160_000_000))
        return first, second

    def test_the_stall_rise_is_divided_by_the_capacity_over_the_span(self):
        # Four cores over twenty seconds permit 8e7 usec, of which 6e7 stalled.
        first, second = self._samples()

        share = throttled_share_over_elapsed(first, second, 20.0)

        assert share == pytest.approx(0.75)

    def test_the_span_length_sets_the_denominator(self):
        first, second = self._samples()

        assert throttled_share_over_elapsed(first, second, 40.0) == pytest.approx(0.375)

    def test_the_span_figure_is_not_the_cumulative_one(self):
        # Against the same samples the cumulative share is 160e6 / (1200 * 4e5),
        # so a figure taken from the counters rather than their rise would be
        # three times smaller and would not move with the span.
        first, second = self._samples()

        assert throttled_share_over_elapsed(first, second, 20.0) != pytest.approx(
            throttled_share(second.stat, second.cpu_max)
        )

    def test_a_sample_that_states_no_period_counter_is_refused(self):
        # A span either side of a group that accounts no period has no rise to
        # measure. Returning zero would assert a span over which nothing was
        # refused, which is a different fact from one that was never measured.
        cpu_max = parse_cpu_max(_QUOTA_ED_MAX)
        first = Sample(cpu_max=cpu_max, stat=_stat())
        second = Sample(
            cpu_max=cpu_max,
            stat=CpuStat(
                usage_usec=2_000_000,
                nr_periods=None,
                nr_throttled=None,
                throttled_usec=None,
            ),
        )

        with pytest.raises(ValueError, match="no period counters"):
            throttled_share_over_elapsed(first, second, 20.0)

    def test_a_span_beginning_in_an_unbounded_group_is_refused(self):
        cpu_max = parse_cpu_max("max 100000")
        first = Sample(cpu_max=cpu_max, stat=_stat())
        second = Sample(cpu_max=cpu_max, stat=_stat())

        with pytest.raises(ValueError, match="no ceiling"):
            throttled_share_over_elapsed(first, second, 20.0)

    def test_a_ceiling_less_group_is_refused_rather_than_measured(self):
        # No cpu.max at all: the group states neither a quota nor a period, so
        # the ceiling's core count the denominator needs does not exist.
        ceiling_less = Sample(
            cpu_max=CpuMax(quota_usec=None, period_usec=None),
            stat=CpuStat(
                usage_usec=401_401,
                nr_periods=None,
                nr_throttled=None,
                throttled_usec=None,
            ),
        )

        with pytest.raises(ValueError, match="no ceiling"):
            throttled_share_over_elapsed(ceiling_less, ceiling_less, 20.0)

    def test_a_span_of_no_length_is_refused(self):
        first, second = self._samples()

        with pytest.raises(ValueError, match="positive"):
            throttled_share_over_elapsed(first, second, 0.0)

    def test_a_counter_going_backwards_is_refused(self):
        first, second = self._samples()

        with pytest.raises(ValueError, match="throttled_usec"):
            throttled_share_over_elapsed(second, first, 20.0)


class TestReadSample:
    def test_reads_both_files_from_the_given_directory(self, tmp_path):
        (tmp_path / "cpu.max").write_text(_QUOTA_ED_MAX)
        (tmp_path / "cpu.stat").write_text(_CPU_STAT_TEXT)

        sample = read_sample(tmp_path)

        assert sample.cpu_max == CpuMax(quota_usec=400_000, period_usec=100_000)
        assert sample.stat.nr_throttled == 250

    def test_a_directory_without_the_files_is_refused(self, tmp_path):
        with pytest.raises(OSError):
            read_sample(tmp_path)


class TestCountersPayload:
    _EXPECTED_KEYS = {
        "usage_usec",
        "nr_periods",
        "nr_throttled",
        "throttled_usec",
        "permitted_usec",
        "throttled_share",
    }

    def test_carries_every_counter_and_the_share_under_a_fixed_key_set(self):
        payload = _counters_payload(
            _stat(nr_periods=1000, throttled_usec=100_000_000),
            parse_cpu_max(_QUOTA_ED_MAX),
        )

        assert set(payload) == self._EXPECTED_KEYS
        assert payload == {
            "usage_usec": 1_000_000,
            "nr_periods": 1000,
            "nr_throttled": 250,
            "throttled_usec": 100_000_000,
            "permitted_usec": 400_000_000,
            "throttled_share": 0.25,
        }

    def test_an_undefined_share_states_its_cause_rather_than_a_null(self):
        # The payload is what a monitor reads, so the distinction has to
        # survive serialisation and not only the in-process return value.
        unbounded = _counters_payload(_stat(), parse_cpu_max("max 100000"))
        unaccounted = _counters_payload(
            _stat(nr_periods=0), parse_cpu_max(_QUOTA_ED_MAX)
        )

        assert unbounded["throttled_share"] == "unbounded"
        assert unbounded["permitted_usec"] is None
        assert unaccounted["throttled_share"] == "no_periods"


class TestParser:
    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            _parser().parse_args([])

    def test_sample_requires_a_directory(self):
        with pytest.raises(SystemExit):
            _parser().parse_args(["sample"])

    def test_interval_takes_a_directory_and_a_float_number_of_seconds(self):
        args = _parser().parse_args(
            ["interval", "--directory", "/sys/fs/cgroup/some.slice", "--seconds", "60"]
        )

        assert args.command == "interval"
        assert args.directory == "/sys/fs/cgroup/some.slice"
        assert args.seconds == 60.0


class TestMain:
    def _write_group(self, directory, cpu_max: str = _QUOTA_ED_MAX) -> None:
        (directory / "cpu.max").write_text(cpu_max)
        (directory / "cpu.stat").write_text(_CPU_STAT_TEXT)

    def test_sample_prints_the_cumulative_reading_as_json(self, tmp_path, capsys):
        self._write_group(tmp_path)

        status = main(["sample", "--directory", str(tmp_path)], _HOST)

        assert status == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"directory", "host", "cpu_max", "cumulative"}
        assert payload["directory"] == str(tmp_path)
        assert payload["cpu_max"] == _QUOTA_ED_MAX
        assert payload["cumulative"]["throttled_share"] == pytest.approx(0.25)

    def test_the_payload_names_the_machine_the_reading_came_from(
        self, tmp_path, capsys
    ):
        # The counter directory resolves on every machine in the cluster with
        # different contents, so without the hostname two readings taken on
        # different computers are indistinguishable and compare as though they
        # described one.
        self._write_group(tmp_path)

        main(["sample", "--directory", str(tmp_path)], _HOST)

        payload = json.loads(capsys.readouterr().out)
        assert payload["host"]["hostname"] == "srv-alpha"

    def test_the_payload_states_which_boot_and_how_long_it_has_run(
        self, tmp_path, capsys
    ):
        # Two readings of cumulative counters can be compared only if a reboot
        # between them is visible: those counters restart at boot, so a share
        # that fell may describe a new boot rather than a recovered machine.
        self._write_group(tmp_path)

        main(["sample", "--directory", str(tmp_path)], _HOST)

        payload = json.loads(capsys.readouterr().out)
        assert payload["host"]["boot_id"] == _HOST.boot_id
        assert payload["host"]["uptime_seconds"] == pytest.approx(_HOST.uptime_seconds)

    def test_an_unbounded_group_serialises_its_share_as_its_cause(
        self, tmp_path, capsys
    ):
        self._write_group(tmp_path, cpu_max="max 100000")

        main(["sample", "--directory", str(tmp_path)], _HOST)

        payload = json.loads(capsys.readouterr().out)
        assert payload["cumulative"]["throttled_share"] == "unbounded"

    def test_interval_prints_the_span_between_two_readings(self, tmp_path, capsys):
        # Zero seconds against an unchanged group: the span is real and empty,
        # which is exactly the reading a caller must be able to tell apart from
        # an unbounded one.
        self._write_group(tmp_path)

        status = main(
            ["interval", "--directory", str(tmp_path), "--seconds", "0"], _HOST
        )

        assert status == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {
            "directory",
            "host",
            "cpu_max",
            "cumulative",
            "interval",
        }
        assert payload["host"]["hostname"] == "srv-alpha"
        interval = payload["interval"]
        assert set(interval) == {
            "usage_usec",
            "nr_periods",
            "nr_throttled",
            "throttled_usec",
            "permitted_usec",
            "throttled_share",
            "seconds",
            "elapsed_seconds",
            "cpu_max_at_end",
            "share_over_elapsed",
        }
        assert interval["seconds"] == 0.0
        assert interval["nr_periods"] == 0
        assert interval["throttled_share"] == "no_periods"
        # No period elapsed, so nothing was refused over the span however
        # short it was — and the span it divided by is stated beside it.
        assert interval["share_over_elapsed"] == 0.0
        assert interval["elapsed_seconds"] >= 0.0
        assert interval["cpu_max_at_end"] == _QUOTA_ED_MAX

    def test_an_interval_over_a_ceiling_less_group_states_the_cause(
        self, tmp_path, capsys
    ):
        # The compute-side group states no ceiling and no period counter, so no
        # span figure exists. The command still emits its reading, stating the
        # cause rather than refusing: it is the side a caller compares against,
        # and a refusal here would leave the sampler unusable where it is most
        # needed.
        (tmp_path / "cpu.stat").write_text(_CPU_STAT_WITHOUT_PERIODS)

        status = main(
            ["interval", "--directory", str(tmp_path), "--seconds", "0"], _HOST
        )

        assert status == 0
        interval = json.loads(capsys.readouterr().out)["interval"]
        assert interval["share_over_elapsed"] == "unbounded"
        assert interval["throttled_share"] == "unbounded"


class TestParseBootIdShape:
    def test_the_kernel_s_own_identifier_is_the_one_accepted(self):
        assert parse_boot_id(_HOST.boot_id) == _HOST.boot_id

    @pytest.mark.parametrize(
        "text",
        [
            "not-a-uuid",
            "5f2c9d841e3a4b709c118a3f2d6b04c7",
            "5F2C9D84-1E3A-4B70-9C11-8A3F2D6B04C7",
            "5f2c9d84-1e3a-4b70-9c11",
            "5f2c9d84-1e3a-4b70-9c11-8a3f2d6b04c7-",
            "5f2c9d84-1e3a-4b70-9c11-8a3f2d6b04cz",
            "00000000-0000-0000-0000-00000000000",
        ],
    )
    def test_a_value_that_is_not_a_boot_identifier_is_refused(self, text):
        # A value merely shaped like one would be reported and compared as
        # though it named a boot, so the accepted set is the kernel's own shape
        # rather than anything non-empty.
        with pytest.raises(ValueError, match="boot identifier"):
            parse_boot_id(text)

    def test_the_refusal_names_the_value_it_read(self):
        with pytest.raises(ValueError, match="not-a-uuid"):
            parse_boot_id("not-a-uuid")


class TestReadHost:
    def _write_identity(self, tmp_path: Path, boot_id: str, uptime: str) -> None:
        (tmp_path / "boot_id").write_text(boot_id)
        (tmp_path / "uptime").write_text(uptime)

    def _point_at(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(throttling, "_BOOT_ID_FILE", str(tmp_path / "boot_id"))
        monkeypatch.setattr(throttling, "_UPTIME_FILE", str(tmp_path / "uptime"))

    def test_reads_the_name_the_boot_and_the_age_of_this_machine(
        self, tmp_path, monkeypatch
    ):
        self._write_identity(tmp_path, f"{_HOST.boot_id}\n", _UPTIME_TEXT)
        self._point_at(tmp_path, monkeypatch)
        monkeypatch.setattr(throttling.socket, "gethostname", lambda: "srv-beta")

        host = read_host()

        assert host.hostname == "srv-beta"
        assert host.boot_id == _HOST.boot_id
        assert host.uptime_seconds == pytest.approx(3_534_804.72)

    def test_a_boot_identifier_file_that_is_not_there_is_refused(
        self, tmp_path, monkeypatch
    ):
        self._write_identity(tmp_path, f"{_HOST.boot_id}\n", _UPTIME_TEXT)
        (tmp_path / "boot_id").unlink()
        self._point_at(tmp_path, monkeypatch)

        with pytest.raises(OSError):
            read_host()

    def test_an_uptime_file_that_is_not_there_is_refused(self, tmp_path, monkeypatch):
        self._write_identity(tmp_path, f"{_HOST.boot_id}\n", _UPTIME_TEXT)
        (tmp_path / "uptime").unlink()
        self._point_at(tmp_path, monkeypatch)

        with pytest.raises(OSError):
            read_host()

    def test_a_boot_identifier_that_is_not_one_is_refused(self, tmp_path, monkeypatch):
        self._write_identity(tmp_path, "not-a-uuid\n", _UPTIME_TEXT)
        self._point_at(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="boot identifier"):
            read_host()

    def test_an_age_that_is_not_a_number_is_refused(self, tmp_path, monkeypatch):
        self._write_identity(tmp_path, f"{_HOST.boot_id}\n", "not-a-number 5.0\n")
        self._point_at(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="uptime"):
            read_host()

    def test_a_hostname_that_cannot_be_read_is_not_swallowed(
        self, tmp_path, monkeypatch
    ):
        # The hostname is what makes two readings comparable, so a machine whose
        # name cannot be read must fail loudly rather than report the reading as
        # unattributed.
        self._write_identity(tmp_path, f"{_HOST.boot_id}\n", _UPTIME_TEXT)
        self._point_at(tmp_path, monkeypatch)

        def _cannot() -> str:
            raise RuntimeError("no hostname")

        monkeypatch.setattr(throttling.socket, "gethostname", _cannot)

        with pytest.raises(RuntimeError, match="no hostname"):
            read_host()


class TestMainReadsTheHostWhenNoneIsGiven:
    def test_the_reading_names_the_machine_the_command_runs_on(
        self, tmp_path, monkeypatch, capsys
    ):
        (tmp_path / "cpu.max").write_text(_QUOTA_ED_MAX)
        (tmp_path / "cpu.stat").write_text(_CPU_STAT_TEXT)
        (tmp_path / "boot_id").write_text(f"{_HOST.boot_id}\n")
        (tmp_path / "uptime").write_text(_UPTIME_TEXT)
        monkeypatch.setattr(throttling, "_BOOT_ID_FILE", str(tmp_path / "boot_id"))
        monkeypatch.setattr(throttling, "_UPTIME_FILE", str(tmp_path / "uptime"))
        monkeypatch.setattr(throttling.socket, "gethostname", lambda: "srv-beta")

        status = main(["sample", "--directory", str(tmp_path)])

        assert status == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["host"]["hostname"] == "srv-beta"
        assert payload["host"]["boot_id"] == _HOST.boot_id
        assert payload["host"]["uptime_seconds"] == pytest.approx(3_534_804.72)


class TestGroupWithNoCeilingFile:
    def _write_ceiling_less_group(self, directory: Path) -> None:
        (directory / "cpu.stat").write_text(_CPU_STAT_WITHOUT_PERIODS)

    def _write_root_group(self, directory: Path) -> None:
        (directory / "cpu.stat").write_text(_CPU_STAT_ROOT_WITHOUT_CEILING)

    def test_the_absence_is_reported_rather_than_raised_on(self, tmp_path):
        self._write_ceiling_less_group(tmp_path)

        sample = read_sample(tmp_path)

        assert sample.cpu_max.can_throttle is False
        assert sample.cpu_max.period_usec is None
        assert sample.stat.usage_usec == 401_401

    def test_the_period_counters_are_absent_rather_than_zero(self, tmp_path):
        # A measured zero would assert that periods elapsed and none was
        # exceeded; here nothing was ever accounted.
        self._write_ceiling_less_group(tmp_path)

        sample = read_sample(tmp_path)

        assert sample.stat.nr_periods is None
        assert sample.stat.nr_throttled is None
        assert sample.stat.throttled_usec is None
        assert sample.stat.period_counters is None

    def test_a_group_with_no_ceiling_still_reports_the_accounted_counters(
        self, tmp_path
    ):
        # The root group declares no ceiling of its own and yet accounts the
        # machine's whole run, so its period counters are genuine values that
        # happen to be zero. Reporting them as absent would discard a count the
        # kernel did keep, which is not the same reading as a group that never
        # accounted one.
        self._write_root_group(tmp_path)

        sample = read_sample(tmp_path)

        assert sample.cpu_max.can_throttle is False
        assert sample.cpu_max.period_usec is None
        assert sample.stat.nr_periods == 0
        assert sample.stat.nr_throttled == 0
        assert sample.stat.throttled_usec == 0
        assert sample.stat.period_counters == (0, 0, 0)

    def test_the_two_no_ceiling_states_are_distinguishable_from_each_other(
        self, tmp_path
    ):
        # An unaccounted group and one accounting zero periods both carry no
        # ceiling, and they are different facts: the first has no count to
        # report, the second a count that reads zero.
        root = tmp_path / "root"
        compute = tmp_path / "compute"
        root.mkdir()
        compute.mkdir()
        self._write_root_group(root)
        self._write_ceiling_less_group(compute)

        accounted = read_sample(root).stat
        unaccounted = read_sample(compute).stat

        assert accounted.period_counters == (0, 0, 0)
        assert unaccounted.period_counters is None
        assert accounted.nr_periods is not None
        assert unaccounted.nr_periods is None
        assert accounted != unaccounted

    def test_both_no_ceiling_states_are_distinguishable_from_a_quota_ed_group(
        self, tmp_path
    ):
        root = tmp_path / "root"
        quota_ed = tmp_path / "quotaed"
        root.mkdir()
        quota_ed.mkdir()
        self._write_root_group(root)
        self._write_quota_ed_group(quota_ed)

        no_ceiling = read_sample(root)
        ceiling = read_sample(quota_ed)

        assert format_cpu_max(no_ceiling.cpu_max) == "absent"
        assert format_cpu_max(ceiling.cpu_max) == _QUOTA_ED_MAX
        assert no_ceiling.cpu_max.can_throttle is False
        assert ceiling.cpu_max.can_throttle is True
        assert no_ceiling.stat.nr_periods == 0
        assert ceiling.stat.nr_periods == 1000

    def _write_quota_ed_group(self, directory: Path) -> None:
        (directory / "cpu.max").write_text(_QUOTA_ED_MAX)
        (directory / "cpu.stat").write_text(_CPU_STAT_TEXT)

    def test_the_ceiling_is_rendered_as_absent_and_not_as_unlimited(self, tmp_path):
        self._write_ceiling_less_group(tmp_path)

        assert format_cpu_max(read_sample(tmp_path).cpu_max) == "absent"
        assert format_cpu_max(parse_cpu_max("max 100000")) == "max 100000"

    def test_the_payload_states_the_absent_counters_as_absent(self, tmp_path, capsys):
        self._write_ceiling_less_group(tmp_path)

        status = main(["sample", "--directory", str(tmp_path)], _HOST)

        assert status == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["cpu_max"] == "absent"
        assert payload["cumulative"]["usage_usec"] == 401_401
        assert payload["cumulative"]["nr_periods"] is None
        assert payload["cumulative"]["permitted_usec"] is None
        assert payload["cumulative"]["throttled_share"] == "unbounded"

    def test_the_payload_states_the_accounted_zeros_as_zeros(self, tmp_path, capsys):
        # The distinction has to survive serialisation and not only the
        # in-process return value, or a monitor reading the JSON sees one
        # missing value for two different states.
        self._write_root_group(tmp_path)

        status = main(["sample", "--directory", str(tmp_path)], _HOST)

        assert status == 0
        cumulative = json.loads(capsys.readouterr().out)["cumulative"]
        assert cumulative["nr_periods"] == 0
        assert cumulative["nr_throttled"] == 0
        assert cumulative["throttled_usec"] == 0
        assert cumulative["permitted_usec"] is None
        assert cumulative["throttled_share"] == "unbounded"

    def test_a_group_that_accounts_no_period_states_no_interval_of_them(self, tmp_path):
        self._write_ceiling_less_group(tmp_path)

        delta = interval_delta(read_sample(tmp_path), read_sample(tmp_path))

        assert delta.usage_usec == 0
        assert delta.period_counters is None

    def test_a_stat_that_states_no_usage_at_all_is_still_refused(self, tmp_path):
        # The usage is what such a group does account, so its absence is
        # malformed text rather than another unaccounted counter, and must not
        # be read as an idle group.
        (tmp_path / "cpu.stat").write_text("user_usec 100\nsystem_usec 50\n")

        with pytest.raises(ValueError, match="usage_usec"):
            read_sample(tmp_path)

    def test_a_stat_that_states_some_period_counters_but_not_all_is_refused(
        self, tmp_path
    ):
        # The kernel writes the three period counters as one account, so a text
        # stating one of them is malformed rather than partly accounted, and
        # neither the count it does state nor an absence may be reported from it.
        (tmp_path / "cpu.stat").write_text("usage_usec 100\nnr_periods 0\n")

        with pytest.raises(ValueError, match="nr_throttled"):
            read_sample(tmp_path)

    def test_a_group_that_loses_a_ceiling_states_no_fabricated_period_counters(
        self, tmp_path
    ):
        # A group moved out from under a ceiling accounts no period, and must
        # not report the earlier counters as a delta it cannot substantiate.
        self._write_ceiling_less_group(tmp_path)
        ceiling_less = read_sample(tmp_path)
        first = Sample(
            cpu_max=parse_cpu_max(_QUOTA_ED_MAX),
            stat=CpuStat(
                usage_usec=100_000,
                nr_periods=1000,
                nr_throttled=250,
                throttled_usec=100_000_000,
            ),
        )

        delta = interval_delta(first, ceiling_less)

        assert delta.period_counters is None
        assert interval_throttled_share(first, ceiling_less) is Unmeasured.UNBOUNDED
