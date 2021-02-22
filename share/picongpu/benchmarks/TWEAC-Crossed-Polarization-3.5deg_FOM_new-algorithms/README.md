Equipping the TWEAC 3.5deg FOM setup with the methods we want to use in TWEAC simulations on Frontier.

Changes in comparison to old setup `TWEAC-Crossed-Polarization-3.5deg_mid-November-FOM-run`:
* `fieldSolver.param`:
  - `CurrentInterpolation = currentInterpolation::None`
  - `maxwellSolver::ArbitraryOrderFDTDPML<4,CurrentInterpolation>`
* `grid.param`: `constexpr float_64 DELTA_T_SI = 6.536577e-18;` (due to AOFDTD solver)
* `species.param`: `particles::pusher::HigueraCary`
* `fieldBackground.param`:
  - include `picongpu/fields/background/templates/twtsfast/twtsfast.hpp`
  - replace namespace `twts` by `twtsfast`
  - remove `w_y` parameter from all twts fields

**Don't forget about the questions in the old setup that (at least partially) still apply to this example!**

