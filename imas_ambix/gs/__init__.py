"""Grad-Shafranov soft-prior support (Stage-2).

The :mod:`imas_ambix.gs` package holds the machine-description geometry and
(later) the Green's-function forward operator that grounds the Stage-1 latent
in raw magnetics.  The geometry table (:mod:`imas_ambix.gs.geometry`) is the
fixed *a-priori* description of the device — sensor positions + orientation,
PF-coil + passive-structure filament geometry, and the limiter contour — keyed
per MAST campaign because the EFIT setup drifts between campaigns.
"""

from __future__ import annotations
