"""Leakage-free Level-2 *input* light-path: read → guard → quantise → store.

This module tokenises **only the provenance-verified L2 inputs** — the
measured observables and the *authorised planned waveforms* — into the
native-cadence v2 token store via the low-rate light path
(:class:`imas_ambix.tokenizer.signals.UniformQuantizer`).  It is the
discriminator counterpart to the camera/world-model encode: it deliberately
does **not** build a second high-frequency patch-transformer corpus (that
duplicates the L1 work the in-flight signal-tokenizer job owns).

The discriminator (reconstruction-vs-plan, keyed on the on-disk uda prefix)
----------------------------------------------------------------------------
* **ENCODE — measured observables** (``input``): coil/solenoid current
  (``AMC_``), gas inboard/outboard/total-injected flows (``AGA_``), summary
  plasma current (``AMC_``) / radiated power (``ABM_``) / NBI power
  (``ANB_``) / neutron rate (``ANU_``), interferometer line density
  (``ANE_``), and the present-when-present soft-X-ray emission cameras
  (``XSX_`` HCAML / HCAMU).
* **ENCODE — authorised planned waveforms** (``planned-action``): the
  pulse-schedule demands (``XDC_`` IPREF / NELREF), the feed-forward coil
  voltage and the gas-valve setpoints (``XDC_``).  These are the *intended*
  trajectory — known a priori, not a reconstruction.
* **REJECT — code-reconstructed state** (never reachable here): equilibrium
  (``EFM_``/``ESM_``), the ESM-derived ``line_average_n_e`` /
  ``greenwald_density`` scalars, and any ``XDC_`` field referencing the
  achieved/reconstructed state (shape / flux-error residuals).

Every field that reaches the quantiser is first passed through the
field-level leakage guard
(:func:`imas_ambix.camdyn.conditioning.assert_no_leakage_fields`), which
keys off the per-field ``uda_name`` and is the gate that *correctly admits*
the planned ``XDC_`` demands while still rejecting reconstructed state and
reconstruction residuals.  (The whole-source guard
:func:`assert_no_leakage_sources` bans the ``XDC`` source group wholesale,
so it is *not* the right gate for a light-path that must carry the planned
waveforms — see the conditioning module docstring.)

Channel naming / de-duplication
-------------------------------
Token channels are named ``"{group}.{var}"`` (and ``"{group}.{var}[{i}]"``
for the i-th column of a multi-channel array), so the three plasma-current
flavours stay distinct: ``summary.ip`` (measured) ≠ ``magnetics.ip``
(measured, different L1 path) ≠ ``pulse_schedule.i_plasma`` (planned
demand).

Store layout
------------
One Zarr group per authorised IMAS group per shot::

    {TOKEN_ROOT}/v2/signals_hf/{shot}/{group}_l2.zarr

The ``_l2`` suffix keeps these groups from ever colliding with the L1
``{group}.zarr`` the running high-frequency encode writes.  Each group is
written at its **native** sampling rate with a per-token time coordinate and
a per-channel validity mask (finite-and-present — never silent zero-fill).
The low-rate light path emits one token per native sample
(``UniformQuantizer.patch_size == 1``), so ``token_rate_hz == native_rate_hz``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.conditioning import assert_no_leakage_fields
from imas_ambix.data.paths import LEVEL2_DIR
from imas_ambix.data.provenance import (
    classify_l2_field,
    is_admissible_input,
    source_of_uda,
)
from imas_ambix.tokenizer.registry import (
    BLOCK_L2_INPUT_LOW,
    L2_BLOCK_VOCAB,
    L2_TIER,
    allocate_l2_input_block,
    build_or_load_v2_registry,
    registry,
)
from imas_ambix.tokenizer.signals import UniformQuantizer
from imas_ambix.tokenizer.store_v2 import (
    STORE_GENERATION,
    StoreV2Attrs,
    save_signal_hf_tokens,
    signal_hf_token_path,
)

logger = logging.getLogger(__name__)

# Token store suffix: the L1 high-frequency encode writes ``{group}.zarr``;
# this light-path writes ``{group}_l2.zarr`` so the two never collide even
# under the same store generation.
L2_GROUP_SUFFIX = "_l2"


def _ensure_l2_block_above_corpus(signals_hf_root=None) -> tuple[int, int]:
    """Seed the singleton registry from on-disk corpus truth + place L2 above it.

    The high-frequency corpus (xma/xim/xsx) is encoded by independent
    processes whose real, model-derived block ranges live only in the on-disk
    store metadata.  This loads (or first-builds + persists) the authoritative
    ``TOKEN_ROOT/v2/registry.json`` manifest, merges the real corpus block
    ranges into the shared singleton, then allocates :data:`BLOCK_L2_INPUT_LOW`
    strictly above the real maximum corpus id.  Idempotent within a process:
    the registry is seeded once and subsequent calls return the same L2 range.

    Returns the ``(start, end)`` global-id range reserved for the L2 input
    block.
    """
    from imas_ambix.data.paths import TOKEN_ROOT
    from imas_ambix.tokenizer.registry import VOCAB_VERSION

    if signals_hf_root is None:
        signals_hf_root = TOKEN_ROOT / VOCAB_VERSION / "signals_hf"

    # Build/load the real corpus namespace and merge it into the singleton so
    # the L2 floor is the REAL corpus maximum (never the old fictional table).
    if not any(
        name in registry._blocks
        for name in (
            "signal_hf_xma_patch_v2",
            "signal_hf_xim_patch_v2",
            "signal_hf_xsx_patch_v2",
        )
    ):
        corpus = build_or_load_v2_registry(signals_hf_root)
        for name, (start, end) in (
            (b.name, (b.start, b.end)) for b in corpus._blocks.values()
        ):
            registry.register_block(name, start, end)

    return allocate_l2_input_block(registry, signals_hf_root=signals_hf_root)


# ---------------------------------------------------------------------------
# Authorised input spec
# ---------------------------------------------------------------------------
#
# The provenance-verified L2 *inputs* — measured observables + authorised
# planned waveforms — keyed by IMAS group → variable names.  This is the
# explicit allow-list of what the light path encodes; it is a strict subset
# of the inventory's ``input`` ∪ ``planned-action`` set (the high-coverage
# actuator / global / emission channels), never the low-coverage profile
# diagnostics that belong to other studies.  Every (group, var) here is
# additionally re-checked at build time against the per-field leakage guard
# and ``is_admissible_input`` — the allow-list is a curation, not a bypass.


@dataclass(frozen=True)
class L2InputSpec:
    """One authorised IMAS group and the input variables to encode from it.

    Attributes
    ----------
    group:
        IMAS group / IDS name (the on-disk Zarr subgroup).
    variables:
        Variable names within the group to encode.  Each may be 1-D
        ``(time,)`` or 2-D ``(channel, time)``; 2-D vars are expanded into
        one token channel per array column.
    required:
        When ``False`` the whole group is treated as present-when-present
        (a missing group / variable is skipped with a log line, not an
        error) — used for the soft-X-ray emission which is absent on many
        shots.
    """

    group: str
    variables: tuple[str, ...]
    required: bool = True


# The authorised input groups.  Ordered measured-first, planned last, with
# the present-when-present emission group flagged ``required=False``.
AUTHORISED_INPUTS: tuple[L2InputSpec, ...] = (
    # --- measured observables -------------------------------------------
    L2InputSpec(
        "pf_active",
        ("coil_current", "solenoid_current", "coil_voltage"),
    ),
    L2InputSpec(
        "gas_injection",
        (
            "inboard_total",
            "outboard_total",
            "total_injected",
            "valve_voltage",
            "valve_target_voltage",
        ),
    ),
    L2InputSpec(
        "summary",
        ("ip", "power_radiated", "power_nbi", "neutron_rates_total"),
    ),
    L2InputSpec("interferometer", ("n_e_line",)),
    # present-when-present soft-X-ray emission cameras (XSX_ HCAML/HCAMU)
    L2InputSpec(
        "soft_x_rays",
        ("horizontal_cam_lower", "horizontal_cam_upper"),
        required=False,
    ),
    # --- authorised planned waveforms (XDC_ demands / setpoints) ---------
    L2InputSpec("pulse_schedule", ("i_plasma", "n_e_line")),
)

# Sanity: no group appears twice (a duplicate would double-write a store).
assert len({s.group for s in AUTHORISED_INPUTS}) == len(AUTHORISED_INPUTS)


# ---------------------------------------------------------------------------
# Per-channel read result
# ---------------------------------------------------------------------------


@dataclass
class L2Channel:
    """One 1-D, time-indexed input channel ready for quantisation.

    Attributes
    ----------
    name:
        Group-qualified, de-duplicated channel name
        (``"{group}.{var}"`` or ``"{group}.{var}[{i}]"``).
    group:
        Source IMAS group.
    var:
        Source variable within the group.
    uda_name:
        The field's ``uda_name`` attribute (drives the leakage guard).
    values:
        ``(n_time,)`` float64 raw values (native cadence, native units).
    valid:
        ``(n_time,)`` bool — finite-and-present mask for this channel.
    units:
        Stored unit string (carried into the store metadata).
    """

    name: str
    group: str
    var: str
    uda_name: str | None
    values: np.ndarray
    valid: np.ndarray
    units: str


@dataclass
class L2GroupRead:
    """All authorised channels read from one group of one shot.

    ``token_time`` is the group's native time coordinate; every channel in
    ``channels`` is aligned to it.  ``native_rate_hz`` is derived from the
    median sample spacing of ``token_time``.
    """

    group: str
    token_time: np.ndarray  # (n_time,) float64
    native_rate_hz: float
    channels: list[L2Channel] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def l2_shot_dir(shot_id: int, level2_dir: Path | str = LEVEL2_DIR) -> Path:
    """Return the ``{shot}.zarr`` directory for a shot under ``level2_dir``."""
    return Path(level2_dir) / f"{shot_id}.zarr"


def _open_group(shot_dir: Path, group: str):  # -> xr.Dataset | None
    """Open one IMAS group of an L2 shot, or ``None`` if it is absent.

    Reads the FAIR-MAST L2 Zarr V3 per-group (``consolidated=False``) with
    xarray — never h5py.  A missing group directory returns ``None`` so the
    caller can presence-guard.
    """
    import xarray as xr

    group_path = shot_dir / group
    if not group_path.exists():
        return None
    try:
        return xr.open_zarr(str(group_path), consolidated=False)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning("could not open %s: %r", group_path, exc)
        return None


def _native_rate_hz(time: np.ndarray) -> float:
    """Median-spacing sampling rate (Hz) of a 1-D time coordinate."""
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    if t.size < 2:
        return 1.0
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return 1.0
    return 1.0 / dt


def _uda_name_of(da) -> str | None:
    """Recover a variable's ``uda_name`` attribute (two written spellings)."""
    uda = da.attrs.get("uda_name")
    if uda is None:
        uda = da.attrs.get("uda")
    return str(uda) if uda is not None else None


def _channels_from_var(group: str, var: str, da) -> list[L2Channel]:
    """Split one (1-D or 2-D) time-indexed variable into 1-D L2Channels.

    A 1-D ``(time,)`` variable becomes one channel ``"{group}.{var}"``.  A
    2-D ``(channel, time)`` variable becomes one channel per column,
    ``"{group}.{var}[{i}]"``.  The time axis is identified as the dimension
    whose name contains ``time`` (the FAIR-MAST groups use ``time``,
    ``time_mirnov``, ``time_saddle``, … per diagnostic).
    """
    dims = tuple(da.dims)
    time_dims = [d for d in dims if "time" in str(d).lower()]
    if not time_dims:
        # No time axis → not a per-shot signal; skip (geometry/infra).
        return []
    time_dim = time_dims[0]
    units = str(da.attrs.get("units", ""))
    uda = _uda_name_of(da)

    # NB: NaNs are PRESERVED in ``values`` (not zero-filled here) so the
    # quantiser's finite-only fit computes per-channel mean/std over the truly
    # valid samples — zero-filling masked positions before the fit corrupts
    # the statistics (a sparse setpoint would gain hundreds of spurious zeros).
    # The quantiser's encode maps any residual NaN to the mid-bin, and the
    # ``valid`` mask is the on-disk source of truth (never silent zero-fill).
    arr = np.asarray(da.values, dtype=np.float64)
    if arr.ndim == 1:
        return [
            L2Channel(
                name=f"{group}.{var}",
                group=group,
                var=var,
                uda_name=uda,
                values=arr,
                valid=np.isfinite(arr),
                units=units,
            )
        ]

    if arr.ndim != 2:
        # Higher-rank (e.g. profile (radius, time)) is out of scope for the
        # light path — those belong to a profile study, not the input set.
        return []

    # 2-D: orient as (channel, time) — the time dim is the longer/explicit one.
    time_axis = dims.index(time_dim)
    if time_axis == 0:
        arr = arr.T  # -> (channel, time)
    n_channels = arr.shape[0]
    out: list[L2Channel] = []
    for i in range(n_channels):
        col = arr[i]
        out.append(
            L2Channel(
                name=f"{group}.{var}[{i}]",
                group=group,
                var=var,
                uda_name=uda,
                values=col,
                valid=np.isfinite(col),
                units=units,
            )
        )
    return out


def read_group(
    shot_id: int,
    spec: L2InputSpec,
    level2_dir: Path | str = LEVEL2_DIR,
) -> L2GroupRead | None:
    """Read every authorised channel of one group for one shot.

    Presence-guards a missing group (returns ``None`` for an optional
    group; raises for a required group only if the group directory is
    entirely absent on a shot where it should exist — callers treat a
    ``None`` from an optional group as "skip").  Each variable is
    leakage-guarded and admissibility-checked here; a variable that would
    not classify as an admissible input is dropped with a log line (it can
    never have reached the allow-list, so this is belt-and-braces).

    All channels are aligned to the group's native ``time`` coordinate.
    Variables that ride a *different* time axis than the group's primary
    time coordinate are kept on their own axis only if every authorised
    variable shares it; mixed-rate groups are split is out of scope — the
    authorised groups here are single-rate (4 kHz controls / 50 kHz SXR).
    """
    shot_dir = l2_shot_dir(shot_id, level2_dir)
    ds = _open_group(shot_dir, spec.group)
    if ds is None:
        if spec.required:
            logger.warning("shot %s: required group %r absent", shot_id, spec.group)
        return None

    # Discover the primary time coordinate of the group (the time dim shared
    # by the authorised variables present).  Use the first present variable.
    present_vars = [v for v in spec.variables if v in ds.data_vars]
    if not present_vars:
        logger.info(
            "shot %s: group %r present but no authorised vars", shot_id, spec.group
        )
        return None

    # Determine the time axis from the first present authorised variable.
    first = ds[present_vars[0]]
    time_dims = [d for d in first.dims if "time" in str(d).lower()]
    if not time_dims:
        logger.warning(
            "shot %s: group %r var %r has no time axis",
            shot_id,
            spec.group,
            present_vars[0],
        )
        return None
    primary_time_dim = time_dims[0]
    token_time = np.asarray(ds[primary_time_dim].values, dtype=np.float64).reshape(-1)

    out = L2GroupRead(
        group=spec.group,
        token_time=token_time,
        native_rate_hz=_native_rate_hz(token_time),
    )

    for var in present_vars:
        da = ds[var]
        uda = _uda_name_of(da)
        # Per-field leakage guard (reconstruction-vs-plan, uda-keyed).  This
        # is the gate that admits planned XDC demands and rejects
        # reconstructed state / residuals.
        assert_no_leakage_fields([(spec.group, var, uda)])
        if not is_admissible_input(spec.group, var, uda):
            fc = classify_l2_field(spec.group, var, uda)
            logger.warning(
                "shot %s: dropping %s.%s (uda=%r) — classified %s (%s)",
                shot_id,
                spec.group,
                var,
                uda,
                fc.classification,
                fc.reason,
            )
            continue

        var_channels = _channels_from_var(spec.group, var, da)
        # Only keep channels that align to the group's primary time axis.
        var_time_dims = [d for d in da.dims if "time" in str(d).lower()]
        if not var_time_dims or var_time_dims[0] != primary_time_dim:
            logger.info(
                "shot %s: %s.%s rides time axis %s != group %s — skipped",
                shot_id,
                spec.group,
                var,
                var_time_dims,
                primary_time_dim,
            )
            continue
        # Length-align defensively to the token_time length.
        for ch in var_channels:
            if ch.values.shape[0] != token_time.shape[0]:
                logger.info(
                    "shot %s: %s length %d != time %d — skipped",
                    shot_id,
                    ch.name,
                    ch.values.shape[0],
                    token_time.shape[0],
                )
                continue
            out.channels.append(ch)

    if not out.channels:
        return None
    return out


# ---------------------------------------------------------------------------
# Quantise + write
# ---------------------------------------------------------------------------


def _group_to_dataset(read: L2GroupRead):  # -> xr.Dataset
    """Build a 1-D, time-indexed ``xr.Dataset`` from a group read.

    Each channel becomes a ``(time,)`` data variable named by its
    de-duplicated channel name — the exact form the UniformQuantizer
    expects (``_time_indexed_channels`` returns 1-D ``time`` vars).
    """
    import xarray as xr

    data_vars = {
        ch.name: (("time",), ch.values.astype(np.float64)) for ch in read.channels
    }
    coords = {"time": read.token_time}
    return xr.Dataset(data_vars=data_vars, coords=coords)


def quantise_group(
    read: L2GroupRead,
    quantizer: UniformQuantizer | None = None,
    *,
    signals_hf_root=None,
    corpus_calibration=None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], UniformQuantizer, tuple[int, int]]:
    """Quantise one group read.

    Calibration mode is chosen by ``corpus_calibration``:

    - **Absolute (``corpus_calibration`` supplied):** the quantizer
      standardises each channel against its CORPUS mean/std, so the same
      physical value maps to the same token in every shot and on every
      machine.  The per-shot ``fit`` is SKIPPED — corpus stats are constant
      across shots.  This is the mode that preserves absolute magnitude.
    - **Per-shot (``corpus_calibration is None``, default):** a per-shot,
      per-group :class:`UniformQuantizer` is fit on this group's own finite
      samples.  Self-calibrating per shot — tight round-trip error, but
      absolute magnitude is NOT comparable across shots.

    Returns ``(tokens, valid, channel_names, quantizer, l2_id_range)`` — the
    ``(n_time, n_channels)`` global token id array, the aligned validity mask,
    the emitted channel order, the quantizer (so the caller can decode for QC),
    and the absolute global-id range ``[start, end)`` the L2 input block
    occupies (recorded in each store).

    The L2 block id range is derived from on-disk corpus truth: it is placed
    strictly ABOVE every id the real xma/xim/xsx stores use (sizes read from
    their ``metadata.codebook_size``), so a decoded L2 id can never alias a
    corpus block.
    """
    ds = _group_to_dataset(read)
    if quantizer is None:
        # Seed the singleton registry from the real on-disk corpus namespace
        # and place the L2 input block STRICTLY ABOVE every real corpus id,
        # BEFORE the UniformQuantizer constructor allocates its name on the
        # shared singleton (idempotent if already placed this process).  A
        # registry re-allocate conflict here is a hard error — never swallowed
        # (a co-resident corpus allocation must not silently skip the L2 group).
        l2_range = _ensure_l2_block_above_corpus(signals_hf_root)
        quant = UniformQuantizer(name=BLOCK_L2_INPUT_LOW, n_bins=L2_BLOCK_VOCAB)
    else:
        quant = quantizer
        l2_range = registry.block_range(quant.name)
    if corpus_calibration is not None:
        # Absolute mode: corpus stats are constant across shots, so skip the
        # per-shot fit entirely and standardise against the corpus mean/std.
        quant.set_calibration(corpus_calibration)
    else:
        quant.fit([ds])
    encoded = quant.encode(ds)
    tokens = np.asarray(encoded.token_ids, dtype=np.int32)
    used = encoded.channel_names

    # Build the validity mask in the emitted channel order.
    valid_by_name = {ch.name: ch.valid for ch in read.channels}
    n_time = tokens.shape[0] if tokens.ndim == 2 else 0
    if not used or n_time == 0:
        return (
            np.zeros((0, 0), dtype=np.int32),
            np.zeros((0, 0), dtype=bool),
            (),
            quant,
            l2_range,
        )
    valid = np.stack([valid_by_name[name] for name in used], axis=-1)
    return tokens, valid, used, quant, l2_range


def build_group(
    shot_id: int,
    spec: L2InputSpec,
    *,
    level2_dir: Path | str = LEVEL2_DIR,
    out_root: Path | str | None = None,
    skip_existing: bool = True,
    signals_hf_root=None,
    corpus_calibration=None,
) -> Path | None:
    """Read → guard → quantise → write one group's L2 input tokens.

    Writes ``{out_root or TOKEN_ROOT}/v2/signals_hf/{shot}/{group}_l2.zarr``.
    Returns the written path, or ``None`` when the group is absent / has no
    authorised channels (present-when-present skip) or when an existing
    store is kept (``skip_existing``).

    ``signals_hf_root`` (default: the real ``TOKEN_ROOT/v2/signals_hf``) is
    the corpus root whose store metadata establishes the real L2 id floor.
    """
    store_group = f"{spec.group}{L2_GROUP_SUFFIX}"
    out_path = _token_path(shot_id, store_group, out_root)
    if skip_existing and out_path.exists():
        # Calibration-mode-aware resume: an absolute re-encode must supersede a
        # legacy per-shot store (different mode → re-encode), while skipping a
        # store already in the requested mode.  A mode read-back failure (e.g.
        # truncated store) falls through to re-encode.
        want_mode = "absolute" if corpus_calibration is not None else "per_shot"
        try:
            import zarr

            existing = dict(zarr.open_group(str(out_path), mode="r").attrs)
            if str(existing.get("calibration_mode", "per_shot")) == want_mode:
                logger.info(
                    "shot %s: %s exists (mode=%s) — skipping",
                    shot_id,
                    out_path.name,
                    want_mode,
                )
                return out_path
        except Exception:  # noqa: BLE001 — unreadable/partial store ⇒ re-encode
            pass

    read = read_group(shot_id, spec, level2_dir)
    if read is None:
        return None

    tokens, valid, used, quant, l2_range = quantise_group(
        read,
        signals_hf_root=signals_hf_root,
        corpus_calibration=corpus_calibration,
    )
    if tokens.size == 0 or not used:
        return None

    units_by_name = {ch.name: ch.units for ch in read.channels}
    uda_by_name = {ch.name: ch.uda_name for ch in read.channels}
    src_by_name = {ch.name: source_of_uda(ch.uda_name) for ch in read.channels}

    t = read.token_time
    window = (float(t[0]), float(t[-1])) if t.size else (0.0, 0.0)

    # The per-channel mean/std the quantiser de-quantises with, plus the
    # calibration provenance, so a reader can tell which mode produced the
    # tokens (absolute = corpus-constant, per-shot = this shot's own stats).
    if corpus_calibration is not None:
        cal_mode = "absolute"
        cal_means = {
            n: float(corpus_calibration[n].mean)
            for n in used
            if n in corpus_calibration
        }
        cal_stds = {
            n: float(corpus_calibration[n].std) for n in used if n in corpus_calibration
        }
    else:
        cal_mode = "per_shot"
        cal_means = {n: quant._means.get(n, 0.0) for n in used}
        cal_stds = {n: quant._stds.get(n, 1.0) for n in used}

    attrs = StoreV2Attrs(
        tokenizer_name=quant.name,
        vocab_version=registry.version,
        native_rate_hz=read.native_rate_hz,
        token_rate_hz=read.native_rate_hz,  # patch_size==1 → 1 token / sample
        n_channels=len(used),
        channel_names=tuple(used),
        phase_preserving=False,  # magnitude-only uniform quantiser
        original_window=window,
        calibration_mode=cal_mode,
        metadata={
            "light_path": "l2_input_low",
            "tier": L2_TIER,
            "n_bins": quant.n_bins,
            "clip_sigma": quant.clip_sigma,
            # Absolute global-id range this L2 store occupies — strictly above
            # every real corpus (xma/xim/xsx) id, so a decoded L2 id never
            # aliases a corpus block.  This is the self-describing leakage
            # contract written into every L2 group.
            "global_id_range": [int(l2_range[0]), int(l2_range[1])],
            # Per-channel mean/std the quantiser actually used: corpus stats
            # in absolute mode, per-shot fitted stats otherwise.  A reader
            # de-quantises with exactly these.
            "channel_means": cal_means,
            "channel_stds": cal_stds,
            "calibration": cal_mode,
            "channel_units": {n: units_by_name.get(n, "") for n in used},
            "channel_uda": {n: uda_by_name.get(n) for n in used},
            "channel_source": {n: src_by_name.get(n) for n in used},
            "imas_group": spec.group,
        },
    )

    return _write_tokens(shot_id, store_group, tokens, t, valid, attrs, out_root)


def build_shot(
    shot_id: int,
    *,
    level2_dir: Path | str = LEVEL2_DIR,
    out_root: Path | str | None = None,
    skip_existing: bool = True,
    specs: tuple[L2InputSpec, ...] = AUTHORISED_INPUTS,
    calibration_by_group: dict[str, dict] | None = None,
) -> dict[str, Path]:
    """Encode every authorised input group for one shot.

    ``calibration_by_group`` maps a group name to its corpus calibration
    (``dict[str, ChannelCalibration]``); a group present here is encoded in
    absolute mode, all others fall back to per-shot fitting.  ``None`` (the
    default) encodes every group per-shot, byte-identical to before.

    Returns a ``{group: written_path}`` map for the groups that produced a
    store (present-when-present groups absent on this shot are simply
    omitted).
    """
    written: dict[str, Path] = {}
    for spec in specs:
        cal = (
            calibration_by_group.get(spec.group)
            if calibration_by_group is not None
            else None
        )
        try:
            path = build_group(
                shot_id,
                spec,
                level2_dir=level2_dir,
                out_root=out_root,
                skip_existing=skip_existing,
                corpus_calibration=cal,
            )
        except ValueError:
            # A registry allocation conflict (the L2 block colliding with a
            # corpus block, or a size mismatch) is a leakage-contract failure,
            # NOT a per-group data hiccup — it must NOT be swallowed, or a
            # co-resident corpus allocation would silently skip the L2 group
            # and the namespace would be wrong on disk.  Propagate and abort.
            logger.error(
                "shot %s group %s: registry allocation error — aborting",
                shot_id,
                spec.group,
            )
            raise
        except Exception:  # noqa: BLE001 — one bad group must not abort the shot
            logger.exception("shot %s group %s failed", shot_id, spec.group)
            continue
        if path is not None:
            written[spec.group] = path
    return written


# ---------------------------------------------------------------------------
# Path / write indirection (so tests can target a temp root, never GPFS)
# ---------------------------------------------------------------------------


def _token_path(shot_id: int, store_group: str, out_root: Path | str | None) -> Path:
    """Token store path, honouring an optional ``out_root`` override."""
    if out_root is None:
        return signal_hf_token_path(shot_id, store_group)
    return (
        Path(out_root)
        / STORE_GENERATION
        / "signals_hf"
        / str(shot_id)
        / f"{store_group}.zarr"
    )


def _write_tokens(
    shot_id: int,
    store_group: str,
    tokens: np.ndarray,
    token_time: np.ndarray,
    valid: np.ndarray,
    attrs: StoreV2Attrs,
    out_root: Path | str | None,
) -> Path:
    """Write a token group either to the default store or under ``out_root``."""
    if out_root is None:
        return save_signal_hf_tokens(
            shot_id, store_group, tokens, token_time, valid, attrs
        )
    # Mirror save_signal_hf_tokens but rooted at out_root (tests / smoke).
    import zarr

    path = _token_path(shot_id, store_group, out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=np.asarray(tokens, dtype=np.int32))
    store.create_array("token_time", data=np.asarray(token_time, dtype=np.float64))
    store.create_array("valid", data=np.asarray(valid, dtype=bool))
    out_attrs = attrs.to_attrs()
    out_attrs.update(
        {"shot_id": int(shot_id), "group": str(store_group), "has_embedding": False}
    )
    store.attrs.update(out_attrs)
    return path


# ---------------------------------------------------------------------------
# CLI entry point (full-corpus driver — used by the sbatch)
# ---------------------------------------------------------------------------


def _enumerate_shots(level2_dir: Path | str) -> list[int]:
    """Every shot id present on disk under ``level2_dir`` (``{id}.zarr``)."""
    out: list[int] = []
    for p in sorted(Path(level2_dir).glob("*.zarr")):
        stem = p.name[: -len(".zarr")]
        if stem.isdigit():
            out.append(int(stem))
    return out


def main(argv: list[str] | None = None) -> int:
    """Full-corpus build driver: encode authorised L2 inputs for every shot.

    ``--shots all`` enumerates every ``{id}.zarr`` under the level-2 dir;
    ``--shots 1,2,3`` takes an explicit list.  ``--skip-existing`` (default)
    makes the run resumable after a time-limit / cancellation.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", default="all", help="'all' or comma-separated ids")
    parser.add_argument("--level2-dir", default=str(LEVEL2_DIR))
    parser.add_argument(
        "--out-root",
        default=None,
        help="token store root override (default: the project TOKEN_ROOT)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="re-encode shots whose store already exists",
    )
    parser.add_argument("--manifest", default=None, help="optional JSON run manifest")
    # Absolute calibration is THE default operative path (a physical value maps
    # to the same token everywhere).  --per-shot is an explicit, clearly-labelled
    # diagnostic escape that z-scores each group per-shot (magnitude NOT
    # comparable across shots).
    parser.add_argument(
        "--per-shot",
        dest="absolute",
        action="store_false",
        help="DIAGNOSTIC ONLY: per-shot z-score each group instead of the "
        "default corpus-calibrated absolute mode (magnitude not preserved)",
    )
    parser.set_defaults(skip_existing=True, absolute=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    calibration_by_group: dict[str, dict] | None = None
    if args.absolute:
        from imas_ambix.calibration.corpus_compute import load_group_calibration

        calibration_by_group = {}
        missing = []
        for spec in AUTHORISED_INPUTS:
            cal = load_group_calibration(spec.group)
            if cal is not None:
                calibration_by_group[spec.group] = cal
                logger.info(
                    "absolute mode: group %r calibration loaded (%d channels)",
                    spec.group,
                    len(cal),
                )
            else:
                missing.append(spec.group)
        if missing:
            # Fail loud — never silently fall back to per-shot in the default
            # absolute path (the repoint contract).  Use --per-shot to opt out.
            raise SystemExit(
                "absolute mode (default) but no corpus calibration for group(s): "
                f"{', '.join(missing)}; run `python -m "
                "imas_ambix.calibration.corpus_compute --group all` first, or "
                "pass --per-shot for an explicit per-shot diagnostic build"
            )

    if args.shots.strip().lower() == "all":
        shots = _enumerate_shots(args.level2_dir)
    else:
        shots = [int(s) for s in args.shots.split(",") if s.strip()]

    logger.info(
        "L2 input encode: %d shots, skip_existing=%s", len(shots), args.skip_existing
    )
    n_groups = 0
    per_shot: dict[int, list[str]] = {}
    for i, shot in enumerate(shots, 1):
        written = build_shot(
            shot,
            level2_dir=args.level2_dir,
            out_root=args.out_root,
            skip_existing=args.skip_existing,
            calibration_by_group=calibration_by_group,
        )
        per_shot[shot] = sorted(written)
        n_groups += len(written)
        if i % 50 == 0 or i == len(shots):
            logger.info(
                "[%d/%d] shot %s groups=%s", i, len(shots), shot, sorted(written)
            )

    if args.manifest:
        import json

        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps(
                {
                    "n_shots": len(shots),
                    "n_groups_written": n_groups,
                    "store_generation": STORE_GENERATION,
                    "per_shot_groups": {str(k): v for k, v in per_shot.items()},
                },
                indent=2,
            )
        )
    logger.info("DONE: %d shots, %d group-stores written", len(shots), n_groups)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
