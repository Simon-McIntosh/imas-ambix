"""Type-6 differential flux loops in the Green's-function forward operator.

A type-6 ``magnetics/flux_loop`` entry carries ``indices_differential`` and no
position: it is the difference of its two named type-1 loops, so its row in
every Green's matrix is the second loop's row minus the first's.  These tests
rebuild a 53-loop store from the merged deck builder, read it through the
catalogue, and pin that identity on the assembled operator, together with the
vacuum fit that scores a differential entry against the difference of the two
measured loops.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.data import geometry_adapter as ga
from imas_ambix.data import selene_deck
from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.data.paths import JT60SA_ROOT
from imas_ambix.gs import operator as op
from imas_ambix.gs.geometry import CircuitDrive, FluxLoop, SensorMapping
from tests.jt60sa_bundle import BUNDLE, SKIP_REASON

pytestmark = pytest.mark.skipif(BUNDLE is None, reason=SKIP_REASON)

DESCRIPTION_ROOT = Path(BUNDLE.store_roots["description"]) if BUNDLE else Path()
STORE_AVAILABLE = BUNDLE is not None and DESCRIPTION_ROOT.is_dir()
requires_store = pytest.mark.skipif(
    not STORE_AVAILABLE,
    reason="the JT-60SA converted description store is not mounted",
)

#: The reference loop the deck differences every other loop against.
REFERENCE = selene_deck.FLUX_LOOP_DIFFERENTIAL_REFERENCE
#: The OP1 shot the rebuilt store serves.
PHASE_SHOT = 101173
#: The no-plasma shot whose measured loop fluxes score the vacuum prediction.
VACUUM_SHOT = 100595
VACUUM_CACHE = JT60SA_ROOT / f"{VACUUM_SHOT}.zarr"
requires_vacuum = pytest.mark.skipif(
    not VACUUM_CACHE.is_dir(),
    reason="the JT-60SA vacuum pulse cache is not mounted",
)


def _catalogue():
    from imas_alambic.machine_map import load_packaged_machine_map

    return load_packaged_machine_map("jt-60sa")


def _provenance(index: int) -> selene_deck.Provenance:
    return selene_deck.Provenance(
        source="magnetics.nc",
        sha256="0" * 64,
        line_start=index + 1,
        line_end=index + 1,
    )


def _declared_sensors():
    """The deck's probes and positioned loops, recovered from a stored IDS.

    The store carries the 17 probes and 27 positioned loops the deck's
    ``geo.in`` declared, whatever else it also holds, so recovering every
    type-1 loop and feeding them to :func:`build_magnetics` regenerates the
    store with exactly the 26 type-6 entries the builder appends.
    """
    import imas

    with imas.DBEntry(
        DESCRIPTION_ROOT / "OP1" / "magnetics.nc",
        "r",
        dd_version=_catalogue().dd_version,
    ) as entry:
        magnetics = entry.get("magnetics", autoconvert=False)
        probes = [
            selene_deck.Sensor(
                r=float(probe.position.r),
                z=float(probe.position.z),
                angle=float(np.rad2deg(probe.poloidal_angle)),
                provenance=_provenance(index),
            )
            for index, probe in enumerate(magnetics.b_field_pol_probe)
        ]
        loops = [
            selene_deck.Sensor(
                r=float(loop.position[0].r),
                z=float(loop.position[0].z),
                angle=None,
                provenance=_provenance(index),
            )
            for index, loop in enumerate(magnetics.flux_loop)
            if int(loop.type.index) == selene_deck.FLUX_LOOP_TYPE_POLOIDAL_FLUX
        ]
    return probes, loops


def _build_53_loop_store(tmp_path: Path) -> Path:
    """A 53-loop OP1 store under ``tmp_path``, built by the merged builder.

    Every file the OP1 binding set reads is copied from the mounted store;
    ``magnetics.nc`` is replaced by one the merged builder writes from the
    deck's probe and loop positions.
    """
    import imas

    destination = tmp_path / "OP1"
    shutil.copytree(DESCRIPTION_ROOT / "OP1", destination)
    probes, loops = _declared_sensors()
    catalogue = _catalogue()
    factory = imas.IDSFactory(catalogue.dd_version)
    geo = selene_deck.GeoIn(
        path="geo.in", sha256="0" * 64, probes=probes, flux_loops=loops
    )
    with imas.DBEntry(
        destination / "magnetics.nc",
        "w",
        dd_version=catalogue.dd_version,
    ) as entry:
        entry.put(selene_deck.build_magnetics(factory, geo))
    return tmp_path


def _differential_rows(table):
    """``(row, first, second)`` for every differential mapping, in row order."""
    return [
        (row, *mapping.indices_differential)
        for row, mapping in enumerate(table.sensor_map)
        if mapping.indices_differential is not None
    ]


def _row_of_loop(table, loop_slot):
    """The sensor row the positioned loop at ``loop_slot`` got."""
    for row, mapping in enumerate(table.sensor_map):
        if (
            mapping.kind == "flux_loop"
            and mapping.indices_differential is None
            and mapping.efm_index == loop_slot
        ):
            return row
    raise AssertionError(f"no positioned row for flux-loop slot {loop_slot}")


@requires_store
def test_rebuilt_store_carries_the_26_differential_entries(tmp_path):
    """The merged builder's store adapts to 27 positioned plus 26 differential.

    Each type-6 entry names the reference loop and one other, in Data Dictionary
    order, carries no position of its own, and is carried by the sensor map with
    the two loops' geometry-table slots as raw DD indices less one.
    """
    _build_53_loop_store(tmp_path)
    table = read_geometry_table(PHASE_SHOT, machine="jt-60sa", store_root=tmp_path)

    differential = [m for m in table.sensor_map if m.indices_differential is not None]
    # 27 positioned loops, plus the 26 type-6 entries the builder appends.
    assert len(table.flux_loops) == 27
    assert len(differential) == 26
    assert [m.indices_differential for m in differential] == [
        (REFERENCE - 1, loop - 1) for loop in range(1, 28) if loop != REFERENCE
    ]
    for mapping in differential:
        assert not np.isfinite(mapping.r) and not np.isfinite(mapping.z)
        assert mapping.flag
    assert [m.amb_channel for m in differential][:2] == [
        f"FL{REFERENCE}-FL1",
        f"FL{REFERENCE}-FL2",
    ]


@requires_store
def test_differential_operator_row_is_second_loop_minus_first(tmp_path):
    """Each type-6 row is its second loop's row minus its first's, in all blocks.

    The operator carries a row per sensor mapping, so a differential entry has
    one: for ``g_pf``, ``g_plasma`` and ``g_passive`` alike it is the difference
    of the two positioned rows its pair names, and the positioned rows keep the
    kernel's own columns, unchanged by the differential rewrite.
    """
    _build_53_loop_store(tmp_path)
    table = read_geometry_table(PHASE_SHOT, machine="jt-60sa", store_root=tmp_path)
    operator = op.build_operator(table)

    pairs = _differential_rows(table)
    assert len(pairs) == 26
    blocks = (operator.g_pf, operator.g_plasma, operator.g_passive)
    for row, first, second in pairs:
        first_row = _row_of_loop(table, first)
        second_row = _row_of_loop(table, second)
        for block in blocks:
            np.testing.assert_allclose(
                block[row, :],
                block[second_row, :] - block[first_row, :],
                rtol=0,
                atol=0,
            )

    positioned = tuple(
        mapping for mapping in table.sensor_map if mapping.indices_differential is None
    )
    bare = op.build_operator(replace(table, sensor_map=positioned))
    for mapping in positioned:
        row = operator.sensor_channels.index(mapping.amb_channel)
        bare_row = bare.sensor_channels.index(mapping.amb_channel)
        for block, reference in zip(
            blocks, (bare.g_pf, bare.g_plasma, bare.g_passive), strict=True
        ):
            np.testing.assert_allclose(
                block[row, :], reference[bare_row, :], rtol=0, atol=0
            )


def test_position_declaration_naming_a_differential_loop_is_refused():
    """A declaration cannot give a type-6 entry the position it does not carry.

    A differential flux loop has no slot in the positioned-loop list, so a
    catalogue declaration that names its channel has nowhere to write: the
    adapter refuses with :class:`GeometryAdapterError` rather than letting the
    mapping's absent index overwrite a positioned loop.  Synthetic on purpose —
    three positioned loops plus one type-6 entry isolate the refusal from the
    mounted store.
    """
    from imas_alambic.machine_map import FluxLoopPositionDeclaration

    loops = [FluxLoop(index=i, r=1.0 + i, z=0.0) for i in range(3)]
    positioned = [
        SensorMapping(
            amb_channel=f"FL{i + 1}",
            kind="flux_loop",
            efm_index=i,
            r=1.0 + i,
            z=0.0,
            angle_deg=None,
            residual_m=0.0,
            flag="",
        )
        for i in range(3)
    ]
    differential = SensorMapping(
        amb_channel="FL7-FL1",
        kind="flux_loop",
        efm_index=-1,
        r=float("nan"),
        z=float("nan"),
        angle_deg=None,
        residual_m=0.0,
        flag="differential flux loop",
        indices_differential=(0, 2),
    )
    declaration = FluxLoopPositionDeclaration(
        name="declares the differential channel",
        acquisition_address="FL7-FL1",
        range_first_shot=PHASE_SHOT,
        range_last_shot=PHASE_SHOT,
        position_verdict="nominal-table",
        declared_r=2.0,
        declared_z=1.0,
        evidence="synthetic three-loop geometry table",
    )
    with pytest.raises(ga.GeometryAdapterError, match="differential flux loop"):
        ga._apply_flux_loop_position_declarations(
            loops, [*positioned, differential], (declaration,)
        )


def _jt60sa_drive_map(catalogue, topology):
    """Materialise the topology's per-circuit drives with their deck ampere-turns."""
    order = tuple(
        dict.fromkeys(
            connection.circuit_identifier for connection in topology.connections
        )
    )
    index = {identifier: position + 1 for position, identifier in enumerate(order)}

    def channel_of(identifier: str) -> str:
        return identifier.split("circuit-", 1)[1].upper().replace("-", "_")

    drives = []
    for identifier in order:
        total = sum(
            connection.turns * connection.current_weight * connection.direction
            for connection in topology.connections
            if connection.circuit_identifier == identifier
        )
        drives.append(
            CircuitDrive(
                circuit=index[identifier],
                channel=channel_of(identifier),
                ampere_turns_per_ampere=total,
                evidence="test drive map",
                conductor=channel_of(identifier),
            )
        )
    return drives


@requires_store
@requires_vacuum
def test_differential_vacuum_prediction_tracks_the_measured_difference():
    """A type-6 entry scores directly against its two loops' measured difference.

    E100595 drives the superconducting coils with no plasma, so each loop
    measures a vacuum prediction.  A pair whose two loops sit close together
    measures a difference dominated by the two channels' common-mode noise, so
    only the pairs whose measured difference is at least twice the reference
    loop's own excursion are scored: both the operator's differential row and
    the measured difference are the second loop minus the first, giving the same
    slope of about -1 the single sized loop shows.
    """
    import zarr

    from imas_alambic.signal_map import load_packaged_signal_map

    catalogue = _catalogue()
    table = read_geometry_table(VACUUM_SHOT, machine="jt-60sa")
    topology = next(
        candidate for candidate in catalogue.drive_topologies if "op1" in candidate.name
    )
    operator = op.build_operator(
        replace(table, circuit_drives=_jt60sa_drive_map(catalogue, topology))
    )

    rules = {
        rule.semantic_id: rule
        for rule in load_packaged_signal_map("jt-60sa", "magnetics").signals
    }
    reference_rule = rules.get(f"magnetics_flux_loop_{REFERENCE}_flux")
    if reference_rule is None:
        pytest.skip("no measured channel for the reference loop")

    store = zarr.open(str(VACUUM_CACHE), mode="r")

    def measured(semantic_id: str):
        rule = rules.get(semantic_id)
        if rule is None:
            return None
        return np.asarray(store[rule.source_group][rule.source_array][:]).ravel()

    measured_reference = measured(f"magnetics_flux_loop_{REFERENCE}_flux")

    coils: dict[str, np.ndarray] = {}
    for rule in load_packaged_signal_map("jt-60sa", "pf_active").signals:
        coil = rule.semantic_id.replace("pf_active_coil_", "").replace("_current", "")
        coils[coil] = np.asarray(store[rule.source_group][rule.source_array][:]).ravel()
    length = min([len(measured_reference), *(len(v) for v in coils.values())])
    current = np.zeros((length, len(operator.pf_amc_channels)))
    for column, channel in enumerate(operator.pf_amc_channels):
        if channel in coils:
            current[:, column] = (
                coils[channel][:length] * operator.pf_current_scales[column]
            )
    measured_reference = measured_reference[:length]

    scored = []
    for row, first, second in _differential_rows(table):
        if first != REFERENCE - 1:
            continue
        measured_loop = measured(f"magnetics_flux_loop_{second + 1}_flux")
        if measured_loop is None:
            continue
        difference = measured_loop[:length] - measured_reference
        if difference.std() < 2 * measured_reference.std():
            continue
        predicted = operator.g_pf[row, :] @ current.T
        correlation = float(np.corrcoef(predicted, difference)[0, 1])
        assert correlation < -0.9, (second + 1, correlation)
        slope = float(np.polyfit(predicted, difference, 1)[0])
        assert slope == pytest.approx(-1.0, abs=0.2), (second + 1, slope)
        scored.append(second + 1)

    assert len(scored) >= 5, f"too few pairs cleared the noise floor: {scored}"
