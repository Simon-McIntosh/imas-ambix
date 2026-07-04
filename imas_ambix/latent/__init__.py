"""The GS-grounded latent engine (v0 stage 2 of the machine-agnostic world model).

This package builds the shared **hybrid latent** (a raw-supervised anchored
block carrying a poloidal-flux ψ representation + a free closure block, in
dimensionless coordinates) and its two physics anchors:

* the **spatial anchor** — an EFIT-free patch-current force-balance substrate
  (:mod:`imas_ambix.latent.patch_basis`, :mod:`imas_ambix.latent.structure_residual`)
  that maps the latent's per-cell patch currents to predicted magnetics at the
  freely-known sensor locations and, differentiably, to the reconstructed
  ψ(R,Z) field.  It is trained against the RAW measured magnetics — never EFIT;
* the **temporal anchor** — a soft, learned flux-diffusion transport prior
  (:mod:`imas_ambix.latent.transport`) on ∂ψ/∂t with strictly-positive learned
  diffusivity (η∥>0 ⇒ D≥0, the arrow of time), learned non-inductive sources
  driven by the command, and sign / dissipation / Volt-second guard-rails.

Field topology — magnetic axis, X-points, LCFS, public/private regions — is a
**deterministic read** of the one solved ψ field
(:mod:`imas_ambix.latent.topology`), never a supervised label; the firewalled
EFIT reconstruction only *scores* that readout.
"""

from __future__ import annotations
