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

import tifffile


def load_membrane(file_in) -> npt.NDArray[any]:
    try:
        segm = tifffile.imread(file_in)
    except:
        raise IOError(f"Error reading in {file_in}. Check file availability or type?")

    return segm


def load_coords(file_in, *, order="zxy") -> npt.NDArray[any]:
    data = np.loadtxt(file_in).astype(int)

    # Reorder coordinates to zxy if needed
    numerical_order = [order.lower().index(i) for i in "zxy"]
    data = data[:, numerical_order]

    return data, numerical_order
