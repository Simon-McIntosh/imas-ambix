# imas-alambic

The slim mint engine: machine maps, signal maps, the transform engine, the
virtual Zarr view, the EDDB read path and the COCOS convention contract.

This distribution exists so the mint can install where `imas-ambix` cannot: the
JT-60SA analysis server runs Python 3.12, while the full world-model stack
requires 3.14 and carries torch, lightning, jax and torax that minting never
touches. The engine's runtime dependencies are `numpy`, `imas-python`, `zarr`,
`pyyaml` and `nova-cocos`.

The engine carries code and schema, never map data. Maps live in a *bundle*: a
directory holding `bundle.json` beside `machine_maps/<machine>.json` and
`maps/<machine>/<ids>.json`. Bundles are found through the
`imas_alambic.bundles` entry-point group (an installed package exposes its own
maps) or a directory named in `IMAS_ALAMBIC_MAP_PATH`.