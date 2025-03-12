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

from itertools import product
from pathlib import Path
from glob import glob

import numpy as np
import numpy.typing as npt
import pandas as pd

import tifffile

from icecream import ic


def parse_membrane_input(path_in: str) -> list:
    # Check if input format correct
    assert (
        Path(path_in).suffix == ".txt" or Path(path_in).is_dir()
    ), "Error in korpus.io:parse_membrane_input: Given input must be either a txt file or a folder."

    # Checks if given path is a txt file
    if Path(path_in).suffix == ".txt":
        with open(Path(path_in)) as f:
            membrane_files = [
                Path(l.rstrip()) for l in f if l.suffix in [".tif", ".tiff"]
            ]
    else:
        membrane_files = glob(f"{path_in}/*.tif") + glob(f"{path_in}/*.tiff")

    return sorted(membrane_files)


def parse_coords_input(path_in: str) -> list:
    # Check if input format correct
    assert (
        Path(path_in).suffix == ".txt" or Path(path_in).is_dir()
    ), "Error in korpus.io:parse_coords_input: Given input must be either a txt file or a folder."

    # Checks if given path is a txt file
    if Path(path_in).suffix == ".txt":
        with open(Path(path_in)) as f:
            coords_files = [Path(l.rstrip()) for l in f if l.suffix == ".txt"]
    else:
        coords_files = glob(f"{path_in}/*.txt")

    return sorted(coords_files)


def load_membrane(file_in: str) -> npt.NDArray[any]:
    try:
        segm = tifffile.imread(file_in)
    except:
        raise IOError(f"Error reading in {file_in}. Check file availability or type?")

    return segm


def load_coords(file_in: str, *, order: str = "zxy") -> npt.NDArray[any]:
    data = np.loadtxt(file_in).astype(int)

    # Reorder coordinates to zxy if needed
    numerical_order = [order.lower().index(i) for i in "zxy"]
    restoration_order = ["zxy".index(i) for i in order.lower()][::-1]
    data = data[:, numerical_order]

    return data, restoration_order


def export_conversion_table(membrane_list: list, coords_list: list) -> pd.DataFrame:
    permutations_gen = product(range(len(membrane_list)), range(len(coords_list)))
    permutations_array = np.array(list(permutations_gen)).T

    membrane_idx = permutations_array[0]
    coords_idx = permutations_array[1]
    membrane_files = [membrane_list[i] for i in membrane_idx]
    coords_files = [coords_list[i] for i in coords_idx]

    data = dict(
        membrane_index=membrane_idx,
        particle_species=coords_idx,
        membrane_file=membrane_files,
        particle_file=coords_files,
    )
    df = pd.DataFrame(data)

    return df
