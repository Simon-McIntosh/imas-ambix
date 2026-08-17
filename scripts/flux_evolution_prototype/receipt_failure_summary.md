# Extraction receipt summary

The bank contains 20 DIII-D train shots and 4,141 labeled frames. The selected set
contains 15 shots with at least 500 ms above 80% of peak plasma current and 12 shots
with at least 200 ms of substantial current ramp, so both ramp and flat-top regimes
are represented directly.

Vacuum Delta-star receipts pass on 4,068 of 4,141 frames (98.24%); every shot has a
majority of passing frames. The 73 failures are concentrated in early current ramp:
the largest contribution is 23 of 59 frames in `d3d_shot_00000a10ac`, all between
180 and 720 ms, with a failed-frame median vacuum/plasma p90 ratio of 0.330. The next
largest contributions are 13 of 122 frames in `d3d_shot_011e881b81` and 12 of 244 in
`d3d_shot_021c290db2`, both confined to 200–440 ms. These frames are retained in the
bank and marked by `vacuum_pass=false`; none were discarded.

Nova's FSA records are well posed on 100% of banked frames. The median within-surface
squared-radius separation fit explains 93.10% of source variation. Across shots, the
median frame-to-frame relative variations are 9.54% for p-prime, 6.49% for FF-prime,
and 1.23% for psi on rho-hat. Three pooled basis components capture 99.76%, 99.85%,
and 100.00% of their respective trajectory variance. The rho-hat coordinate remains a
geometric proxy because the corpus omits the boundary toroidal-field calibration; it
must not be reported as exact normalized toroidal flux.

The numerical distribution and representative trajectory are shown in
`receipt_and_basis_summary.png` and `representative_trajectories.png`.
