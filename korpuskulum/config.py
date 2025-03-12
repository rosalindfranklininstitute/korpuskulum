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

import typing

from korpuskulum import objects


def objectify_user_input(
    pixel_size_nm: typing.Optional[float],
    dist_range: typing.Optional[list],
    coords_files: typing.Optional[list],
    membrane_files: typing.Optional[list],
    order: typing.Optional[str],
) -> objects.Config:
    """Objectifying user-provided input as a Config object

    Args:
    pixel_size_nm (float) : Pixel size of tomogram in nanometers
    dist_range (list)     : List of floats indicating the range of accepted particle-membrane distances: [min, max]
    coords_files (list)   : List of files containing coordinates of picked particles
    membrane_files (list) : List of files containing 3D maps of segmented membranes
    order (str)           : Order of coordinates in which particle coordinates are represented

    Returns:
    Config
    """
    params = objects.Config()
    for key in params.__dict__.keys():
        params.__setattr__(key, locals()[key])

    return params
