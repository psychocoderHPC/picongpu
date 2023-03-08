import openpmd_api as api
import numpy as np
import math
import json
import argparse
from pathlib import Path
import scipy.constants as cs

# we work with PIConGPU coordinates: x,y,z
# so that the openPMD array needs to be rearanged first

def get_beam_unit_vectors(side_str):
    str_to_z_dir = {'x': (1, 0, 0), 'xr': (-1, 0, 0),
                    'y': (0, 1, 0), 'yr': (0, -1, 0),
                    'z': (0, 0, 1), 'zr': (0, 0, -1)}
    
    str_to_y_dir = {'x': (0, 1, 0), 'xr': (0, -1, 0),
                    'y': (-1, 0, 0), 'yr': (1, 0, 0),
                    'z': (-1, 0, 0), 'zr': (-1, 0, 0)}
    
    str_to_x_dir = {'x': (0, 0, -1), 'xr': (0, 0, -1),
                    'y': (0, 0, -1), 'yr': (0, 0, -1),
                    'z': (0, 1, 0), 'zr': (0, -1, 0)}
    z = np.array(str_to_z_dir[side_str])
    z = z / np.linalg.norm(z)
    x = np.array(str_to_x_dir[side_str])
    x = x / np.linalg.norm(x)
    y = np.array(str_to_y_dir[side_str])
    y = y / np.linalg.norm(y)

    # check orthogonality
    assert np.dot(x, y) == 0
    assert np.dot(x, z) == 0
    # check right-handedness
    assert np.dot(np.cross(x ,y), z) > 0
    return x, y, z


def get_beam_origin(side_str, sim_box_size, offset_pic):
    str_to_origin = {'x': (0, 0.5, 0.5), 'xr': (1.0, 0.5, 0.5),
                     'y': (0.5, 0, 0.5), 'yr': (0.5, 1, 0.5),
                     'z': (0.5, 0.5, 0), 'zr': (0.5, 0.5, 1)}
    return sim_box_size * np.array(str_to_origin[side_str]) + offset_pic


def transfer_coordinates(v, x_hat, y_hat, z_hat):
    return np.array((np.dot(v, x_hat), np.dot(v, y_hat), np.dot(v, z_hat)))


def get_beam_coordinate_system(side_str, offset_beam, sim_box_size):
    x_beam, y_beam, z_beam = get_beam_unit_vectors(side_str)
    x_pic = transfer_coordinates(np.array((1, 0, 0)), x_beam, y_beam, z_beam)
    y_pic = transfer_coordinates(np.array((0, 1, 0)), x_beam, y_beam, z_beam)
    z_pic = transfer_coordinates(np.array((0, 0, 1)), x_beam, y_beam, z_beam)
    offset_beam = np.array([offset_beam[0], offset_beam[1], 0])
    offset_pic = transfer_coordinates(offset_beam, x_pic, y_pic, z_pic)
    origin = get_beam_origin(side_str, sim_box_size, offset_pic)
    return origin, x_beam, y_beam, z_beam


class BeamCoordinates:
    def __init__(self, side_str, offset_beam, sim_box_size):
        args = get_beam_coordinate_system(side_str, offset_beam, sim_box_size)
        self.origin, self.x_beam, self.y_beam, self.z_beam = args

    def __call__(self, x, y, z):
        x -= self.origin[0]
        y -= self.origin[1]
        z -= self.origin[2]
        x_b = x * self.x_beam[0] + y * self.x_beam[1] + z * self.x_beam[2]
        y_b = x * self.y_beam[0] + y * self.y_beam[1] + z * self.y_beam[2]
        z_b = x * self.z_beam[0] + y * self.z_beam[1] + z * self.z_beam[2]
        return x_b, y_b, z_b



def gaussian(x, sigma):
    tmp = x / sigma
    exponent = -0.5 * (tmp * tmp)
    return math.exp(exponent)

class GaussianProfile:
    def __init__(self, sigma_x, sigma_y):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, x, y):
        x_term = (x / self.sigma_x)**2
        y_term = (y / self.sigma_y)**2
        exponent = -0.5 * (x_term + y_term)
        return np.exp(exponent)


class ConstShape:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, t):
        mask = np.where((t >= self.start_time)  & (t< self.end_time))
        factor = np.zeros_like(t)
        factor[mask]=1
        return factor


class GaussianShape(ConstShape):
    def __init__(self, start_time, end_time, FWHM, cut_time_front, cut_time_back):
        super().__init__(start_time, end_time)
        self.FWHM = FWHM
        self.cut_time_front = cut_time_front
        self.cut_time_back = cut_time_back

    def __call__(self, t):
        raise NotImplemented
        factor = super().__call__(t)
        return factor

def generate_intensity_array(sim_shape, beam_profile, beam_shape, coor_trans, delta_x, delta_y, delta_z, t):
    x_pic = np.arange(sim_shape[0]) + 0.5
    y_pic = np.arange(sim_shape[1]) + 0.5
    z_pic = np.arange(sim_shape[2]) + 0.5

    x_pic *= delta_x
    y_pic *= delta_y
    z_pic *= delta_z
    x_beam, y_beam, z_beam = coor_trans(x_pic[:, None, None], y_pic[None, :, None], z_pic[None, None, :])
    return beam_profile(x_beam, y_beam) * beam_shape(t - z_beam / cs.c)


def verify_results(picongpu_run_dir, side_str, offset):
    time_step = 1.0e-6 / cs.c
    cell_size = 3.0e-6
    sigma = (10, 7)
    start_time = 0  # time_steps
    end_time = 20  # time_steps
    profile = GaussianProfile(sigma[0] * cell_size, sigma[1] * cell_size)
    shape = ConstShape(time_step * start_time, time_step * end_time)
    sim_shape = (24, 24, 24)
    coor_trans = BeamCoordinates(side_str, (offset[0] * cell_size,
                                            offset[1] * cell_size),
                                 np.array(sim_shape) * cell_size)

    factor = 1e5  # max photon count per cell

    # load simulation output
    infix = "_%06T"
    backend = "bp"
    mesh_name = "ph_all_particleCounter"
    openpmd_path = picongpu_run_dir / 'openPMD'
    series_path_str = "{}/{}{}.{}".format(str(openpmd_path),
                                          'simData',
                                          infix, backend)
    series = api.Series(series_path_str, api.Access.read_only)

    checks = [[], []]
    for iteration in series.read_iterations():
        mesh = iteration.meshes[mesh_name]
        mrc = mesh[api.Mesh_Record_Component.SCALAR]
        data = mrc.load_chunk()
        series.flush()
        data *= mrc.unit_SI
        data = data.T  # z,y,x -> x,y,z
        axis_map = {'x': 0, 'xr': 0, 'y': 1, 'yr': 1, 'z': 2, 'zr': 2}
        reverse_map = {'x': False, 'xr': True, 'y': False,
                       'yr': True, 'z': False, 'zr': True}
        prop_axis_idx = axis_map[side_str]

        reduced_cells_shape = list(sim_shape)
        reduced_cells_shape[prop_axis_idx] *= 3
        reduced_cells_shape = tuple(reduced_cells_shape)

        reduced_cell_size = [cell_size, cell_size, cell_size]
        reduced_cell_size[prop_axis_idx] /= 3
        reduced_cell_size = tuple(reduced_cell_size)

        particle_count = generate_intensity_array(reduced_cells_shape, profile,
                                                  shape, coor_trans,
                                                  *reduced_cell_size,
                                                  (iteration.iteration_index
                                                   - 1) * time_step)

        class SwapIdx:
            def __init__(self, idx1, idx2):
                self.idx1 = idx1
                self.idx2 = idx2

            def __getitem__(self, slicing):
                slicing = list(slicing)
                slicing[self.idx1], slicing[self.idx2] =\
                    slicing[self.idx2], slicing[self.idx1]
                return tuple(slicing)
        swap = SwapIdx(0, prop_axis_idx)

        if reverse_map[side_str]:
            particle_count[swap[:-1, :, :]] = particle_count[swap[1:, :, :]]
            particle_count[swap[-1, :, :]] = 0.0
        else:
            particle_count[swap[1:, :, :]] = particle_count[swap[:-1, :, :]]
            particle_count[swap[0, :, :]] = 0.0

        particle_count = particle_count[swap[0::3, :, :]] / 3 +\
                         particle_count[swap[1::3, :, :]] / 3 +\
                         particle_count[swap[2::3, :, :]] / 3
        particle_count *= factor
        check = np.all(np.isclose(data,
                                  particle_count))
        checks[0].append(check)
        checks[1].append(iteration.iteration_index)
    return checks

def main():
    parser = argparse.ArgumentParser(description="Verify an ExternalBeam test")
    parser.add_argument('--dir', nargs='?',
                        help="the simOutput directory",
                        default=Path.cwd())
    args = parser.parse_args()
    picongpu_run_dir = Path(args.dir)
    with open(picongpu_run_dir / "test_setup.json", "r") as f:
        parameters = json.load(f)
    checks = verify_results(picongpu_run_dir, **parameters)
    for ii, iteration in enumerate(checks[1]):
        print(fr"iteration: {iteration}, "
              fr"{'passed' if checks[0][ii] else 'failed'} \n")
    assert np.all(checks[0])


if __name__ == '__main__':
    main()
