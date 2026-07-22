"""Toroidal sensor positions from the MAST L2 magnetics group.

The frozen ``efm`` static-setup geometry the equilibrium engine reads
(:mod:`imas_ambix.gs.geometry`) is purely axisymmetric: each magnetic sensor
has an ``(R, Z)`` and (for a B-probe) a poloidal pickup angle, but **no toroidal
position**.  For an n != 0 applied field (the EFCC / ELM coils) the absolute
response of a probe depends on where it sits toroidally, so the axisymmetric
table cannot resolve the per-probe coupling — only a phase-independent envelope.

The MAST **L2** magnetics group *does* carry the toroidal coordinate.  This
module reads it — the only place toroidal sensor geometry enters the engine's
world.  It is apparatus metadata (sensor placement), never a reconstruction
output, so it is outside the leakage firewall exactly like ``(R, Z)``.

What the L2 group stores
------------------------
* **Poloidal probes** (``obv``/``obr``/``ccbv``/``cc``/``omv``):
  ``b_field_pol_probe_<fam>_r/_z`` per geometry channel, and toroidal angle as
  either a single ``_phi`` array or a *pair* of bank arrays ``_phi_1``/``_phi_2``.
  The ``obv``/``obr``/``ccbv`` arrays sit at two toroidal banks (150 deg and
  330 deg); the L2 metadata lists both candidate banks per geometry channel and
  does not pin which physical acquisition channel lives in which bank — so both
  are returned and a downstream validation resolves the assignment from the
  measured coupling.
* **Toroidal saddle loops** (``b_field_tor_probe_saddle_*``): 28-vertex polygons
  spanning ~330 deg of arc; the representative toroidal position is the circular
  mean of the polygon vertices.
* **Toroidal probe array** (``b_field_tor_probe_cc``): 36 probes at 12 evenly
  spaced sectors, geometry only.

Firewall: reads only the ``magnetics`` apparatus geometry (R, Z, phi) — no
equilibrium, boundary, psi, or reconstruction output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: poloidal-probe families the L2 group tabulates a toroidal position for.
POL_PROBE_FAMILIES = ("obv", "obr", "ccbv", "cc", "omv")


def _circular_mean_deg(phi_deg: np.ndarray) -> float:
    """Circular mean of angles [deg] (handles the 0/360 seam)."""
    rad = np.deg2rad(np.asarray(phi_deg, dtype=np.float64).reshape(-1))
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


@dataclass(frozen=True)
class ProbeToroidal:
    """Toroidal geometry of one poloidal-probe family (per geometry channel)."""

    family: str
    geometry_channels: tuple[str, ...]
    r: np.ndarray  # (n,) major radius [m]
    z: np.ndarray  # (n,) height [m]
    banks_deg: tuple[float, ...]  # candidate toroidal angles [deg]

    def channel_index(self, channel: str) -> int | None:
        """Index of an amb/geometry channel name within this family (or None)."""
        key = _norm_channel(channel)
        for i, gc in enumerate(self.geometry_channels):
            if _norm_channel(gc) == key:
                return i
        return None


@dataclass(frozen=True)
class SaddleToroidal:
    """Toroidal geometry of the saddle detector loops."""

    band: str
    channels: tuple[str, ...]
    r: np.ndarray  # (n,)
    z: np.ndarray  # (n,)
    phi_deg: np.ndarray  # (n,) circular-mean toroidal position [deg]


@dataclass
class ToroidalGeometry:
    """The toroidal sensor geometry available for one shot."""

    shot: int
    probes: dict[str, ProbeToroidal] = field(default_factory=dict)
    saddle: SaddleToroidal | None = None

    def summary(self) -> dict:
        return {
            "shot": self.shot,
            "probe_families": {
                fam: {"n": len(p.geometry_channels), "banks_deg": list(p.banks_deg)}
                for fam, p in self.probes.items()
            },
            "n_saddle": 0 if self.saddle is None else len(self.saddle.channels),
        }


def _norm_channel(name: str) -> str:
    """Lower-case, strip separators and an ``AMB_`` prefix for name matching."""
    s = str(name).lower().replace("amb_", "")
    for sep in (" ", "_", "-", "/", "."):
        s = s.replace(sep, "")
    return s


def _open_l2_magnetics(shot: int):
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415

    path = LEVEL2_DIR / f"{int(shot)}.zarr"
    if not path.exists():
        return None
    try:
        return zarr.open_group(str(path), mode="r")["magnetics"]
    except Exception:  # noqa: BLE001
        return None


def _bank_angles(grp, keys, fam: str) -> tuple[float, ...]:
    """Distinct candidate toroidal-bank angles [deg] for a probe family."""
    banks: list[float] = []
    for suffix in ("_phi", "_phi_1", "_phi_2"):
        k = f"b_field_pol_probe_{fam}{suffix}"
        if k in keys:
            vals = np.asarray(grp[k], dtype=np.float64).reshape(-1)
            if vals.size:
                # a bank array is (near-)constant across channels; store its
                # per-array representative angle(s)
                for a in np.unique(np.round(vals, 1)):
                    banks.append(float(a))
    return tuple(sorted(set(banks)))


def read_probe_toroidal(shot: int) -> dict[str, ProbeToroidal]:
    """Per-family toroidal geometry of the poloidal probes from L2 (or empty)."""
    grp = _open_l2_magnetics(shot)
    if grp is None:
        return {}
    keys = set(grp.array_keys())
    out: dict[str, ProbeToroidal] = {}
    for fam in POL_PROBE_FAMILIES:
        rk, zk = f"b_field_pol_probe_{fam}_r", f"b_field_pol_probe_{fam}_z"
        if rk not in keys or zk not in keys:
            continue
        r = np.asarray(grp[rk], dtype=np.float64).reshape(-1)
        z = np.asarray(grp[zk], dtype=np.float64).reshape(-1)
        gck = f"b_field_pol_probe_{fam}_geometry_channel"
        if gck in keys:
            gchans = tuple(str(x) for x in np.asarray(grp[gck]).reshape(-1))
        else:
            gchans = tuple(f"{fam}{i + 1:02d}" for i in range(r.size))
        banks = _bank_angles(grp, keys, fam)
        out[fam] = ProbeToroidal(
            family=fam,
            geometry_channels=gchans[: r.size],
            r=r,
            z=z,
            banks_deg=banks,
        )
    return out


def read_saddle_toroidal(shot: int, *, band: str = "saddle_m") -> SaddleToroidal | None:
    """Toroidal geometry of the saddle detector loops from L2 (or None)."""
    grp = _open_l2_magnetics(shot)
    if grp is None:
        return None
    keys = set(grp.array_keys())
    pk = f"b_field_tor_probe_{band}_phi"
    rk = f"b_field_tor_probe_{band}_r"
    zk = f"b_field_tor_probe_{band}_z"
    if not ({pk, rk, zk} <= keys):
        return None
    phi = np.asarray(grp[pk], dtype=np.float64)  # (n, n_vertex) deg
    r = np.asarray(grp[rk], dtype=np.float64)
    z = np.asarray(grp[zk], dtype=np.float64)
    n = phi.shape[0]
    ck = f"b_field_tor_probe_{band}_geometry_channel"
    chans = (
        tuple(str(x) for x in np.asarray(grp[ck]).reshape(-1))
        if ck in keys
        else tuple(f"{band}{i:02d}" for i in range(n))
    )
    return SaddleToroidal(
        band=band,
        channels=chans[:n],
        r=np.array([float(np.mean(r[i])) for i in range(n)]),
        z=np.array([float(np.mean(z[i])) for i in range(n)]),
        phi_deg=np.array([_circular_mean_deg(phi[i]) for i in range(n)]),
    )


def read_toroidal_geometry(shot: int) -> ToroidalGeometry:
    """Assemble the toroidal sensor geometry available for ``shot``."""
    return ToroidalGeometry(
        shot=int(shot),
        probes=read_probe_toroidal(shot),
        saddle=read_saddle_toroidal(shot),
    )


__all__ = [
    "POL_PROBE_FAMILIES",
    "ProbeToroidal",
    "SaddleToroidal",
    "ToroidalGeometry",
    "read_probe_toroidal",
    "read_saddle_toroidal",
    "read_toroidal_geometry",
]
