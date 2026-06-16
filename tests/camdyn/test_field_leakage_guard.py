"""Field-level leakage guard on the L2 conditioning path.

``assert_no_leakage_fields`` is the corrected, field-level guard the L2
loader uses.  It must:

* AUTHORISE planned pulse-schedule waveforms (demanded Ip/density,
  feed-forward coil voltage, gas valve demands) — these were wrongly
  banned by the source-group guard.
* REJECT code-reconstructed equilibrium (EFM_/ESM_) and the
  reconstruction-derived summary scalars.
* Leave the legacy source-group guard semantics
  (``assert_no_leakage_sources`` / ``BANNED_CONDITIONING_SOURCES``)
  unchanged for existing camera-frame conditioning callers.
"""

from __future__ import annotations

import pytest

from imas_ambix.camdyn.conditioning import (
    BANNED_CONDITIONING_SOURCES,
    assert_no_leakage_fields,
    assert_no_leakage_sources,
)


def test_legacy_source_guard_semantics_unchanged():
    # The source-group guard still bans efm/esm/xdc wholesale for the
    # camera-frame loader — additive extension must not alter this.
    assert frozenset({"efm", "esm", "xdc"}) == BANNED_CONDITIONING_SOURCES
    for bad in ("efm", "esm", "xdc"):
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage_sources(["amc", bad])
    # measured-only set passes
    assert_no_leakage_sources(["amc", "anb", "aga", "ane"])  # no raise


def test_field_guard_authorises_planned_actions():
    # All three planned-action families must pass the field-level guard.
    planned = [
        ("pulse_schedule", "i_plasma", "XDC_IP_T_IPREF"),
        ("pulse_schedule", "n_e_line", "XDC_DENSITY_T_NELREF"),
        ("pf_active", "coil_voltage", "XDC_PF_F_P1"),
        ("gas_injection", "valve_voltage", "XDC_GAS_F_G1"),
        ("gas_injection", "valve_target_voltage", "XDC_GAS_T_G1"),
    ]
    assert_no_leakage_fields(planned)  # no raise


def test_field_guard_admits_planned_alongside_measured():
    fields = [
        ("magnetics", "ip", "AMC_PLASMA CURRENT"),
        ("interferometer", "n_e_line", "ANE_DENSITY"),
        ("gas_injection", "inboard_total", "AGA_INBOARD_TOTAL"),
        ("pf_active", "coil_voltage", "XDC_PF_F_P1"),  # planned, authorised
        ("pulse_schedule", "i_plasma", "XDC_IP_T_IPREF"),  # planned
    ]
    assert_no_leakage_fields(fields)  # no raise


def test_field_guard_rejects_reconstructed_equilibrium():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage_fields(
            [
                ("magnetics", "ip", "AMC_PLASMA CURRENT"),  # ok
                ("equilibrium", "psi", "EFM_PSI(R,Z)"),  # banned
            ]
        )


def test_field_guard_rejects_reconstruction_derived_scalars():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage_fields([("summary", "line_average_n_e", "ESM_NE_BAR")])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage_fields([("summary", "greenwald_density", "ESM_N_GREENWALD")])


def test_field_guard_rejects_xdc_reconstruction_residual():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage_fields(
            [("pulse_schedule", "shape_error", "XDC_SHAPE_FLUXERR")]
        )


def test_field_guard_does_not_raise_on_probe_or_infra():
    # Probe target (Dα) and infra (geometry) are not leakage — they do not
    # raise (selecting them as inputs is governed separately).
    assert_no_leakage_fields(
        [
            (
                "spectrometer_visible",
                "filter_spectrometer_dalpha_voltage",
                "XIM_DA/HM10/R",
            ),
            ("wall", "limiter_r", None),
        ]
    )  # no raise
