"""Tests for the CPU throttling sampler over cgroup v2 text."""

from __future__ import annotations

import pytest

from imas_ambix.agent.throttling import (
    CpuMax,
    CpuStat,
    Sample,
    format_cpu_max,
    interval_delta,
    interval_throttled_share,
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

        assert throttled_share(stat, parse_cpu_max("max 100000")) is None

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

        assert throttled_share(stat, parse_cpu_max(_QUOTA_ED_MAX)) is None


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

        assert interval_throttled_share(first, second) is None


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
