"""Tests for the CPU throttling sampler over cgroup v2 text."""

from __future__ import annotations

import json

import pytest

from imas_ambix.agent.throttling import (
    CpuMax,
    CpuStat,
    Sample,
    Unmeasured,
    _counters_payload,
    _parser,
    format_cpu_max,
    interval_delta,
    interval_throttled_share,
    main,
    parse_cpu_max,
    parse_cpu_stat,
    permitted_thread_usec,
    read_sample,
    throttled_share,
)

# The account slice on the login node: four cores out of each 100 ms period.
_QUOTA_ED_MAX = "400000 100000"

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

        status = main(["sample", "--directory", str(tmp_path)])

        assert status == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"directory", "cpu_max", "cumulative"}
        assert payload["directory"] == str(tmp_path)
        assert payload["cpu_max"] == _QUOTA_ED_MAX
        assert payload["cumulative"]["throttled_share"] == pytest.approx(0.25)

    def test_an_unbounded_group_serialises_its_share_as_its_cause(
        self, tmp_path, capsys
    ):
        self._write_group(tmp_path, cpu_max="max 100000")

        main(["sample", "--directory", str(tmp_path)])

        payload = json.loads(capsys.readouterr().out)
        assert payload["cumulative"]["throttled_share"] == "unbounded"

    def test_interval_prints_the_span_between_two_readings(self, tmp_path, capsys):
        # Zero seconds against an unchanged group: the span is real and empty,
        # which is exactly the reading a caller must be able to tell apart from
        # an unbounded one.
        self._write_group(tmp_path)

        status = main(["interval", "--directory", str(tmp_path), "--seconds", "0"])

        assert status == 0
        payload = json.loads(capsys.readouterr().out)
        interval = payload["interval"]
        assert set(interval) == {
            "usage_usec",
            "nr_periods",
            "nr_throttled",
            "throttled_usec",
            "permitted_usec",
            "throttled_share",
            "seconds",
            "cpu_max_at_end",
        }
        assert interval["seconds"] == 0.0
        assert interval["nr_periods"] == 0
        assert interval["throttled_share"] == "no_periods"
        assert interval["cpu_max_at_end"] == _QUOTA_ED_MAX
