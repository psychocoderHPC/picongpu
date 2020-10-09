#pragma once

namespace picongpu
{

    DEF_PARAMETER(float_64, DELTA_T, "sim.delta_t", "simulation time step");
    PARAM_ADD(float_X, DELTA_T, PIC);


    DEF_PARAMETER(float_64, CELL_WIDTH, "sim.cellSize_x", "cell size in x");
    PARAM_ADD(float_X, CELL_WIDTH, PIC);

    DEF_PARAMETER(float_64, CELL_HEIGHT, "sim.cellSize_y", "cell size in y");
    PARAM_ADD(float_X, CELL_HEIGHT, PIC);

    DEF_PARAMETER(float_64, CELL_DEPTH, "sim.cellSize_z", "cell size in z");
    PARAM_ADD(float_X, CELL_DEPTH, PIC);

} // namespace picongpu
