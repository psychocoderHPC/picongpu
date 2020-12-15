#pragma once

namespace picongpu
{

    CREATE_GLOBAL_VAR(float_X, DELTA_T, PIC);
    DEF_PARAMETER(float_64, DELTA_T, "sim.delta_t", "simulation time step", units::second);


    CREATE_GLOBAL_VAR(float_X, CELL_WIDTH, PIC);
    DEF_PARAMETER(float_64, CELL_WIDTH, "sim.cellSize_x", "cell size in x", units::meter);


    CREATE_GLOBAL_VAR(float_X, CELL_HEIGHT, PIC);
    DEF_PARAMETER(float_64, CELL_HEIGHT, "sim.cellSize_y", "cell size in y", units::meter);

    CREATE_GLOBAL_VAR(float_X, CELL_DEPTH, PIC);
    DEF_PARAMETER(float_64, CELL_DEPTH, "sim.cellSize_z", "cell size in z", units::meter);

} // namespace picongpu
