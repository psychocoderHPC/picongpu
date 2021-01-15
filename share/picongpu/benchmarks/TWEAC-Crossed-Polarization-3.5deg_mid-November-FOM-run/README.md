# Questions/Notes for the setup

* Use binomial current interpolation? Currently activated.

* Use `CurrentSolver::EmZ`? Currently activated. Michael says yes. Publication still to be done!

* Use `particles::shapes::PCS`? Currently activated. A higher one is available (`P4S`) but is way slower at negligible accuracy increase, according to @psychocoderHPC.

* Use `particles::pusher::Vay`? Currently activated. @Klaus: What is the exact difference to `Boris`?

* Use `maxwellSolver::YeePML`? Currently just `maxwellSolver::Yee` due to memory footprint.

* Need to further increase `exchangeMemCfg` for electrons? Currently the value is already larger than the standard.

* Tweak minimum particle weighting of `0.001`? We expect 40 real e- per cell at the base density and want to initialize 50 macroparticles per cell.

* Initialize electrons with constant drift velocity and temperature? Currently activated. We do this to trigger a calculation of the electron current in the FOM run. Otherwise electrons do not move initially and the current calculation is omitted.

* We do not want to give gas macro particles a temperature in moving window simulations as this leads to density extrema at GPU boundaries

* Use `startPosition::Random`? Currently activated. May save some time at startup when defining in-cell particle positions beforehand.



# Things to change for TWEAC production run

## In source code

* Implement `SuperPush` for the integration of the background Field. @psychocoderHPC, @sbastrakov

* Implement parameter struct for TWTS laser pulse in order to avoid paramter editing in four pulses.

* Deactivate constant initial electron drift.

* Deactivate initial temperature.

* Use `maxwellSolver::YeePML`.

* Use the real density profile in `density.param` and remove the `REL_DENSITY_HIGH` factor in the return statement of `FreeFormulaFunctor`.

* Increase the laser a_0 to real value.

* Set the laser focus position in y and time delay to a sensible value.

* Adjust the start of the moving window accordingly to synchronize moving window with laser propagation.

* Increase typical particles per cell to 50, or higher?

* Once implemented, use Higuera pusher.


## In `*.cfg`

* Remove periodic boundaries

* Activate moving window

* Increase number of time steps

* Increase run time

* Add plugins (counter, diagnostics, output, checkpoints,...)
