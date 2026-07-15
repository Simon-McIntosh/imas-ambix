# Soft-prior interior solve — shared module contracts (equilibrium-boundary-closure §3)

Working reference for the fleet building the soft-prior force-balance solve. NOT a plan
doc; a coherence anchor for interfaces. Conventions are binding across every module.

## Binding physics conventions (all modules)
- **TOTAL flux** Φ = 2π R A_φ [Wb]. Every Green's column carries total flux.
- Source: Δ*Φ = −2π μ0 R jφ  (the 2π is real — see gs_solve.py:517).
- MAST sign: positive Ip ⇒ magnetic axis is a **maximum** of ψ, so ψ_axis > ψ_boundary.
- Normalised flux ψ_N = (ψ − ψ_axis)/(ψ_bnd − ψ_axis): 0 at axis, 1 at boundary.
- Current basis (`gs_solve.profile_basis`): jφ = (R/R0)·Σ cᵖ_k φ_k(ψ_N)  +  (R0/R)·Σ cᶠ_k φ_k(ψ_N).
  The R/R0 family is the pressure-gradient (p′) drive; the R0/R family the FF′ drive.
- Coil/passive couplings ALWAYS `imas_ambix.gs.cylinder.hybrid_greens` (finite-area
  cylinder) → (psi, br, bz). NEVER point-filament.
- Machine-agnostic: every knob dimensionless or geometry-scaled. NEVER fixed metres.
- EFIT firewalled: referee/scoring only, never an input or target in any fit path.

## The per-sweep LSQ variable vector (gs_solve.solve_equilibrium_lsq)
`x = [coeffs (k_dof = n_p+n_f), a_pass (kp)]`, optionally extended by a rank-1 gauge
offset `g`. Data rows (whitened) `b_mat@coeffs + bp@a_pass = y`; Ip hard KKT
`a_anchor·coeffs = Ip` (passive excluded); smoothness Gram on coeffs; passive ridge.
Every prior below contributes EXTRA weighted rows on this same `x` (the orchestrator
wires them in; modules assemble rows given inputs passed to them).

## Module ownership (file scopes — exclusive)
- boundary_harmonic.py + scripts/harmonic_prior_freeze.py + test_harmonic_gauge.py → Worker A
- imas_ambix/latent/boundary_prior.py + test_boundary_prior.py → Worker B
- imas_ambix/latent/profile_regularization.py + test_profile_regularization.py → Worker C
- imas_ambix/latent/moment_priors.py + test_moment_priors.py → Worker D
- gs_solve.py, closure_gate_eval.py, evidence doc/figures, plan → orchestrator

## Firewall-safe data actually available (payload / corpus)
SlicePayload: measured/vacuum/mask/scale (magnetics), i_pf [A], ip_amperes [A].
Anchored raw scalars: **Ip (Rogowski), n_e (line-averaged density)** ONLY.
NO diamagnetic loop, NO Thomson Te, NO li/βp scalars. Consequence: βp+li/2 is
derivable from external magnetics (firewall-safe); full p′/ff′ separation from an
independent *temperature* is DATA-GATED — build machinery + synthetic test, record gate.
