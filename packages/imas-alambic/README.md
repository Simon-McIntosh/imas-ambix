# imas-alambic

The slim write engine: machine maps, signal maps, the transform engine, the
virtual Zarr view, the EDDB read path and the COCOS convention contract.

This distribution exists so the writer can install where `imas-ambix` cannot: the
JT-60SA analysis server runs Python 3.12, while the full world-model stack
requires 3.14 and carries torch, lightning, jax and torax that writing never
touches. The engine's runtime dependencies are `numpy`, `imas-python`, `zarr`,
`pyyaml` and `nova-cocos`.

The engine carries code and schema, never map data. Maps live in a *bundle*: a
directory holding `bundle.json` beside `machine_maps/<machine>.json` and
`maps/<machine>/<ids>.json`. Bundles are found through the
`imas_alambic.bundles` entry-point group (an installed package exposes its own
maps) or a directory named in the resolved map search path.

Settings — the map search path, the IDS root, the cache and the machine — are
resolved by `imas_alambic.settings`, at the highest precedence that names each:
a command-line flag, its `IMAS_ALAMBIC_*` variable, or a default derived from
`IMAS_ALAMBIC_HOME` (`maps/current` and `ids`). The cache falls back from the
bundle's `eddb_cache` store role to `~/.cache/imas-alambic`, and the machine is
inferred when exactly one is reachable. `imas-alambic config` prints every
resolved setting with its source.