#   Copyright 2025 Rosalind Franklin Institute
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
import numpy.typing as npt

from sklearn.metrics import pairwise_distances as PD


def get_distribution(seg_map: npt.NDArray[any],
                     coords: npt.NDArray[any],
                     pixel_size_nm: float,
                     *,
                     slice_idx: list=[],
) -> list:
    """Evaluates the distribution of particles for given slices.
    If slice indices are not given, evaluate the entire stack.

    Args:
    seg_map (ndarray)          : 3D map containing one segmented membrane
    coords (ndarray)           : Coordinates of the picked particles in the ZXY order
    pixel_size_nm (float)      : Pixel size of seg_map in nanometers
    slice_idx (Optional, list) : List of Z-slice indices to be evaluated

    Returns:
    list
    """

    full_distro_list = []
    if len(slice_idx)==0:
        slice_idx = range(len(seg_map))
    else:
        for slice_no in slice_idx:
            seg_mask = np.argwhere(seg_map[slice_no]==1)
            trimmed_coords_slice = np.asarray([i for i in coords if i[0]==slice_no])
            coords_slice_2d = trimmed_coords_slice[:, [2, 1]]

            try:
                dmat = PD(seg_mask, coords_slice_2d)
            except:
                continue
            else:
                min_dist = dmat.min(axis=0)
                closest_args = np.argmin(dmat, axis=0)
                distribution = (coords_slice_2d - seg_mask[closest_args]) * pixel_size_nm
                slice_list = [slice_no] * len(distribution)

                mask_fit_slope = np.polyfit(*seg_mask.T, deg=1)[0]
                mask_fit_normal = np.array([-mask_fit_slope, 1])
                if mask_fit_normal[1] > 0:
                    mask_fit_normal *= -1

                orientations = (np.sum(distribution*mask_fit_normal, axis=1) >= 0).astype(int)

                full_distro_list.append(
                    (distribution, slice_list, orientations, trimmed_coords_slice)
                )

    return full_distro_list
