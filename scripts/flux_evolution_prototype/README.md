# Flux-trajectory extraction spike

This throwaway measurement lane selects twenty DIII-D train shots using plasma-current
coverage, applies a second-order rectilinear Delta-star stencil to every labeled total-flux
map, and separates the source within each normalized-flux annulus by its squared-radius
dependence. The resulting p-prime, FF-prime, surface-mean toroidal current, flux trajectory,
Nova flux-surface averages, and vacuum receipt are banked per shot as compressed NumPy files.

The challenge labels do not provide a boundary toroidal-field calibration. The banked
`rho_hat` coordinate is therefore a geometric normalized-toroidal-flux proxy, not an exact
toroidal-flux coordinate. `kernel_gaps.json` records this and the other missing Nova seams.

Run with the project environment and Nova source tree visible:

```bash
python pull_regime_subset.py SOURCE_DIR SELECTION_JSON --count 20
PYTHONPATH=/path/to/nova python extract_flux_trajectories.py \
  SELECTION_JSON BANK_DIR SUMMARY_DIR
```
