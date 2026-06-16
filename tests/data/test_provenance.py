"""Field-level provenance resolver + reconstruction-vs-plan classifier.

These tests pin the lead-corrected principle:

* PLANNED actions are AUTHORISED — pulse_schedule demanded Ip/density,
  the pf_active feed-forward coil voltage (``XDC_PF_F``), and the
  gas-injection valve demands (``XDC_GAS_*``) all classify as
  ``planned-action`` and are admissible inputs.
* Code-reconstructed state and reconstruction-derived scalars are BANNED
  — the EFIT/Solov'ev equilibrium (``EFM_``/``ESM_``) and the
  ESM-derived ``summary.line_average_n_e`` (``ESM_NE_BAR``) +
  ``summary.greenwald_density`` (``ESM_N_GREENWALD``).
* Measured diagnostics (``AMC_``/``ANE_``/``AGA_``/...) are admissible
  inputs.
"""

from __future__ import annotations

import pytest

from imas_ambix.data.provenance import (
    BANNED,
    INFRA,
    INPUT,
    PLANNED_ACTION,
    PROBE_TARGET,
    UDA_SOURCE_MAP,
    classify_l2_field,
    is_admissible_input,
    source_of_uda,
)

# ---------------------------------------------------------------------------
# uda_name → source prefix resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uda", "expected"),
    [
        # underscore form
        ("EFM_PSI(R,Z)", "EFM"),
        ("EFM_PLASMA_CURR(R,Z)", "EFM"),
        ("ESM_NE_BAR", "ESM"),
        ("AMC_PLASMA CURRENT", "AMC"),  # whitespace after prefix
        ("AMC_P2IL FEED CURRENT", "AMC"),
        ("ANE_DENSITY", "ANE"),
        ("ABM_PRAD_POL", "ABM"),
        ("AGA_INBOARD_TOTAL", "AGA"),
        ("ANB_TOT_SUM_POWER", "ANB"),
        ("ANU_NEUTRONS", "ANU"),
        ("ADG_DENSITY_GRADIENT", "ADG"),
        ("XSX_HCAML#1", "XSX"),  # '#' separator
        ("XIM_DA/HM10/R", "XIM"),  # '/' inside underscore form
        ("XDC_IP_T_IPREF", "XDC"),
        ("XDC_GAS_F_G1", "XDC"),
        ("XDC_PF_F_P1", "XDC"),
        # slash / path form
        ("/xdc/gas/f/tc5a", "XDC"),
        ("/xdc/ip/t/ipref", "XDC"),
        ("/xsx/HCAM/L/1", "XSX"),
        ("/xmo/OMAHA/1LZ", "XMO"),
        ("xbt/channel01", "XBT"),  # lowercase, no leading slash
        # missing / empty
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_source_of_uda(uda, expected):
    assert source_of_uda(uda) == expected


def test_uda_source_map_covers_expected_systems():
    # the prompt's prefix map is honoured
    assert UDA_SOURCE_MAP["AMC"] == "magnetics"
    assert UDA_SOURCE_MAP["AMB"] == "magnetics"
    assert UDA_SOURCE_MAP["AMA"] == "magnetics"
    assert UDA_SOURCE_MAP["ANE"] == "interferometer"
    assert UDA_SOURCE_MAP["ABM"] == "bolometer"
    assert UDA_SOURCE_MAP["AGA"] == "gas_injection"
    assert UDA_SOURCE_MAP["XSX"] == "soft_x_rays"
    assert UDA_SOURCE_MAP["XIM"] == "spectrometer_visible"
    assert UDA_SOURCE_MAP["ANB"] == "nbi"
    assert UDA_SOURCE_MAP["ANU"] == "neutron_diagnostic"
    assert UDA_SOURCE_MAP["ADG"] == "spectrometer_visible"
    assert UDA_SOURCE_MAP["EFM"] == "equilibrium_efit"
    assert UDA_SOURCE_MAP["ESM"] == "equilibrium_solovev"
    assert UDA_SOURCE_MAP["XDC"] == "pulse_schedule"


# ---------------------------------------------------------------------------
# PLANNED ACTIONS — authorised (the corrected principle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "var", "uda"),
    [
        # pulse_schedule demanded Ip / density
        ("pulse_schedule", "i_plasma", "XDC_IP_T_IPREF"),
        ("pulse_schedule", "n_e_line", "XDC_DENSITY_T_NELREF"),
        ("pulse_schedule", "i_plasma", "/xdc/ip/t/ipref"),
        ("pulse_schedule", "n_e_line", "/xdc/density/t/nelref"),
        # pf_active feed-forward coil voltage (XDC_PF_F)
        ("pf_active", "coil_voltage", "XDC_PF_F_P1"),
        ("pf_active", "coil_voltage", "/xdc/pf/f/p1"),
        # gas-injection valve demands (XDC_GAS_*)
        ("gas_injection", "valve_voltage", "XDC_GAS_F_G1"),
        ("gas_injection", "valve_target_voltage", "XDC_GAS_T_G1"),
        ("gas_injection", "valve_voltage", "/xdc/gas/f/tc5a"),
        ("gas_injection", "valve_target_voltage", "/xdc/gas/t/g1"),
    ],
)
def test_planned_actions_are_authorised(group, var, uda):
    fc = classify_l2_field(group, var, uda)
    assert fc.classification == PLANNED_ACTION, fc
    assert fc.source == "XDC"
    assert not fc.review
    assert is_admissible_input(group, var, uda)


# ---------------------------------------------------------------------------
# BANNED — reconstructed state + reconstruction-derived scalars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "var", "uda"),
    [
        # EFIT equilibrium fields
        ("equilibrium", "psi", "EFM_PSI(R,Z)"),
        ("equilibrium", "j_phi", "EFM_PLASMA_CURR(R,Z)"),
        ("equilibrium", "q95", "EFM_Q_95"),
        ("equilibrium", "lcfs_r", "EFM_LCFS(R)_(C)"),
        ("equilibrium", "magnetic_axis_r", "EFM_MAGNETIC_AXIS_R"),
        ("equilibrium", "li", "EFM_LI"),
        ("equilibrium", "wmhd", "EFM_WPLASMD"),
        # Solov'ev equilibrium fields
        ("equilibrium", "vloop_static", "ESM_V_LOOP_STATIC"),
        ("equilibrium", "vloop_dynamic", "ESM_V_LOOP_DYNAMIC"),
        # ESM-derived summary scalars (embed the EFIT boundary)
        ("summary", "line_average_n_e", "ESM_NE_BAR"),
        ("summary", "greenwald_density", "ESM_N_GREENWALD"),
    ],
)
def test_reconstructed_and_derived_are_banned(group, var, uda):
    fc = classify_l2_field(group, var, uda)
    assert fc.classification == BANNED, fc
    assert not is_admissible_input(group, var, uda)


def test_derived_scalars_banned_even_if_source_relabelled():
    # The explicit field list bans the derived scalars regardless of the
    # uda prefix — a future relabelling to a measured source must not
    # silently re-admit them.
    fc = classify_l2_field("summary", "line_average_n_e", "AMC_FOO")
    assert fc.classification == BANNED
    fc = classify_l2_field("summary", "greenwald_density", "ANE_BAR")
    assert fc.classification == BANNED


def test_xdc_reconstruction_residual_is_banned_and_flagged():
    # A hypothetical XDC field encoding an error against the achieved
    # (reconstructed) shape/flux → banned even though source is XDC.
    for uda in (
        "XDC_SHAPE_FLUXERR",
        "XDC_IP_ERROR",
        "/xdc/shape/residual",
        "XDC_BOUNDARY_DEVIATION",
    ):
        fc = classify_l2_field("pulse_schedule", "shape_error", uda)
        assert fc.classification == BANNED, fc
        assert not is_admissible_input("pulse_schedule", "shape_error", uda)


def test_ambiguous_xdc_defaults_to_banned_for_review():
    # An XDC field with neither a demand marker nor a residual marker is
    # ambiguous → default BANNED + review flag (fail safe).
    fc = classify_l2_field("pulse_schedule", "mystery_field", "XDC_MYSTERY")
    assert fc.classification == BANNED
    assert fc.review is True


def test_unknown_source_defaults_to_banned_for_review():
    fc = classify_l2_field("some_group", "some_var", "ZZZ_WHATEVER")
    assert fc.classification == BANNED
    assert fc.review is True
    assert fc.source == "ZZZ"


# ---------------------------------------------------------------------------
# MEASURED inputs — admissible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "var", "uda", "system"),
    [
        ("magnetics", "ip", "AMC_PLASMA CURRENT", "magnetics"),
        ("magnetics", "b_field_pol_probe_obr_field", "AMB_OBR01", "magnetics"),
        ("magnetics", "flux_loop_flux", "AMB_FL/CC03", "magnetics"),
        ("magnetics", "b_field_tor_probe_saddle_field", "ASM_SAD/M01", "magnetics"),
        ("interferometer", "n_e_line", "ANE_DENSITY", "interferometer"),
        ("gas_injection", "inboard_total", "AGA_INBOARD_TOTAL", "gas_injection"),
        ("gas_injection", "outboard_total", "AGA_OUTBOARD_TOTAL", "gas_injection"),
        ("pf_active", "coil_current", "AMC_P2IL FEED CURRENT", "magnetics"),
        ("pf_active", "solenoid_current", "AMC_SOL CURRENT", "magnetics"),
        ("summary", "ip", "AMC_PLASMA CURRENT", "magnetics"),
        ("summary", "power_radiated", "ABM_PRAD_POL", "bolometer"),
        ("summary", "power_nbi", "ANB_TOT_SUM_POWER", "nbi"),
        ("summary", "neutron_rates_total", "ANU_NEUTRONS", "neutron_diagnostic"),
        ("soft_x_rays", "horizontal_cam_lower", "XSX_HCAML#1", "soft_x_rays"),
        ("soft_x_rays", "horizontal_cam_upper", "XSX_HCAMU#1", "soft_x_rays"),
        (
            "spectrometer_visible",
            "density_gradient",
            "ADG_DENSITY_GRADIENT",
            "spectrometer_visible",
        ),
    ],
)
def test_measured_fields_are_inputs(group, var, uda, system):
    fc = classify_l2_field(group, var, uda)
    assert fc.classification == INPUT, fc
    assert fc.level1_system == system
    assert is_admissible_input(group, var, uda)


# ---------------------------------------------------------------------------
# PROBE TARGET — Dα, default-off (not banned, not a default input)
# ---------------------------------------------------------------------------


def test_dalpha_is_probe_target_not_input():
    fc = classify_l2_field(
        "spectrometer_visible",
        "filter_spectrometer_dalpha_voltage",
        "XIM_DA/HM10/R",
    )
    assert fc.classification == PROBE_TARGET
    # not banned, but also NOT a default admissible input
    assert not is_admissible_input(
        "spectrometer_visible",
        "filter_spectrometer_dalpha_voltage",
        "XIM_DA/HM10/R",
    )


# ---------------------------------------------------------------------------
# INFRA — geometry / static machine description (no uda_name)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "var"),
    [
        ("magnetics", "flux_loop_r"),
        ("pf_active", "p5_upper_r"),
        ("pf_passive", "ring_r"),
        ("wall", "limiter_r"),
        ("soft_x_rays", "tangential_cam_origin_r"),
    ],
)
def test_geometry_fields_are_infra(group, var):
    fc = classify_l2_field(group, var, None)
    assert fc.classification == INFRA
    assert fc.source is None
    assert not is_admissible_input(group, var, None)
