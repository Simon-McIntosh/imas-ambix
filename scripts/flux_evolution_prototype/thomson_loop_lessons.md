# Frozen-geometry Thomson loop lessons

This demo substitutes a labeled-map/FSA forward for a free-boundary solve. It
therefore measures reduced flux-function tracking under fixed labeled geometry;
it does not demonstrate boundary, q95, or beta-N recovery.

- Keep: one cross-shot transport closure, plasma current solely as a transport
  boundary condition, a joint p-prime/FF-prime evolution basis trained away
  from the scored shot, row-whitened observable SVD, and explicit per-channel
  innovation receipts.
- Change: the production smoother needs a free-boundary solve, uncertainty in
  density and electron-to-total-pressure conversion, time-varying FSA geometry,
  and additional observations that directly constrain FF-prime. The present
  FF-prime correction is covariance-mediated because the Thomson surrogate has
  no direct FF-prime sensitivity.
- Drop: claims that Thomson-only frozen geometry identifies LCFS, q95, beta-N,
  or hidden FF-prime directions. No Thomson channel was dropped; uncertain
  instances were retained with widened sigma.
- SVD assessment: not adequate as a Thomson-only production basis.
  Rank 6 retains 100.00% of the initial
  whitened observable energy; 72.7% of posterior
  channels pass the 5% whiteness test, and the update changes joint NRMSE by
  +2.15% relative to its prior.

The reduced update changes joint tracking error by +2.15% relative to the transport prior and accounts for 26.88% of the total gain over persistence. For FF-prime alone it changes error by -0.02% and accounts for -0.12% of total FF-prime gain.
