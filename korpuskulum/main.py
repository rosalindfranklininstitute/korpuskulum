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


import os
from pathlib import Path
import typing
from typing_extensions import Annotated

from datetime import datetime as dt
import logging
import re

import typer
import numpy as np
import pandas as pd
import starfile

from icecream import ic

from korpuskulum import (config, io, evaluate, plotting, prog_bar)


VERSION = "0.1.0"

def callback():
    pass


def unique_rows(array_in):
    sorted_idx = np.lexsort(array_in.T)
    sorted_array = array_in[sorted_idx, ...]
    row_mask = np.append([True], np.any(np.diff(sorted_array, axis=0), 1))
    out = sorted_array[row_mask]

    return out


app = typer.Typer(callback=callback)

@app.callback()
def callback(
        dev_mode: Annotated[
            bool,
            typer.Option(
                "-d", "--dev",
                help="Developer mode. If true, verbose error messages and tracebacks (with full printout of local variables) will be enabled.",)
        ] = False,
):
    app.pretty_exceptions_show_locals = dev_mode


@app.command()
def main(
        membrane_input: Annotated[
            typing.Optional[str],
            typer.Option(
                "-m", "--membranes",
                help="Input file(s) for membranes. Can be either a single txt file or a directory. If the input is a txt file, the file must contain the paths to all the membrane segmentation maps as TIFFs. If the input is a folder, Korpuskulum will perform evaluations on all TIFF files in the given folder."),
        ] = None,
        coords_input: Annotated[
            typing.Optional[str],
            typer.Option(
                "-c", "--coords",
                help="Input file(s) for particle coordinates. Can be either a single txt file or a directory. If the input is a txt file, the file must contain the paths to all the particle coordinates as txt files. If the input is a folder, Korpuskulum will perform evaluations on all txt files in the given folder."),
        ] = None,
        pixel_size_nm: Annotated[
            typing.Optional[float],
            typer.Option(
                "-s", "--pixel_size",
                help="Pixel size of tomogram(s) in nanometers."
            ),
        ] = None,
        dist_range: Annotated[
            list[float, float],
            typer.Option(
                "-r", "--range",
                help="Range of accepted particle-membrane distances."
            ),
        ] = [2, 10],
        coords_order: Annotated[
            typing.Optional[str],
            typer.Option(
                "-o", "--order",
                help="Order of coordinate system used in the particle coordinates. Korpuskulum uses the order ZXY and will internally convert the system to this order if the user specifies otherwise. Outputs will be in the same order as the input system. (Optional; case-insensitive)"),
        ] = "zxy",
        output_folder: Annotated[
            typing.Optional[str],
            typer.Option(
                "-out", "--output",
                help="Path to output folder. If specified folder does not exist, Korpuskulum will create it first. Default: ./results/"),
        ] = "./results/",
):
    """Main API for Korpuskulum"""

    # Check if parameters given
    assert membrane_input is not None, \
        "A file/folder must be specified for the --membranes parameter."
    assert coords_input is not None, \
        "A file/folder must be specified for the --coords parameter."
    assert pixel_size_nm is not None, \
        "A value must be given to the --pixel_size parameter."
    
    # Parse and convert user inputs
    membrane_list = io.parse_membrane_input(membrane_input)
    coords_list = io.parse_coords_input(coords_input)
    
    # Objectify user inputs
    params = config.objectify_user_input(
        pixel_size_nm=pixel_size_nm,
        dist_range=dist_range,
        coords_files=coords_list,
        membrane_files=membrane_list,
        order=coords_order,
    )
    
    # Evaluation loops
    with prog_bar.prog_bar as p:
        prog_bar.clear_tasks(p)
        for (m_idx, m) in p.track(enumerate(membrane_list),
                                  total=len(membrane_list)
        ):
            seg_map = io.load_membrane(m)
            seg_nonempty = np.argwhere(np.sum(seg_map, axis=(1, 2))!=0).flatten()

            for (c_idx, c) in enumerate(coords_list):
                coords, restoration_order = io.load_coords(c, order=params.order)

                # Calculate distributions
                eval_slice_idx = np.intersect1d(seg_nonempty,
                                                np.unique(coords.T[0]).astype(int))
                stack_distro_list = evaluate.get_distribution(
                    seg_map=seg_map,
                    coords=coords,
                    pixel_size_nm=pixel_size_nm,
                    slice_idx=eval_slice_idx
                )
                stack_distro = np.vstack([i[0] for i in stack_distro_list])
                slice_numbers = np.concatenate([i[1] for i in stack_distro_list])
                orientations = np.concatenate([i[2] for i in stack_distro_list])
                trimmed_coords = np.vstack([i[3] for i in stack_distro_list])

                # Get minimum distance and angular arguments in radians
                min_dist = np.linalg.norm(stack_distro, axis=1)
                angles = np.arctan2(*stack_distro.T[::-1])

                # Plot polar distribution and minimum distance distribution
                if not Path(output_folder).is_dir():
                    Path(output_folder).mkdir()
                if not Path(f"{output_folder}/coords/").is_dir():
                    Path(f"{output_folder}/coords/").mkdir()
                file_prefix = f"ptcl_{c_idx:02}_memb_{m_idx:02}"

                plotting.plot_polar_hist(
                    dist_array=min_dist,
                    angle_array=angles,
                    savefig=f"{output_folder}/{file_prefix}_polar_distro.png"
                )
                plotting.plot_min_dist_hist(
                    dist_array=min_dist,
                    orientations=orientations,
                    protein_name=f"ptcl_{c_idx}",
                    membrane_name=f"memb_{m_idx}",
                    dist_low=min(params.dist_range),
                    dist_high=max(params.dist_range),
                    savefig=f"{output_folder}/{file_prefix}_mindist_distro.png"
                )

                # Pick coordinates for configuration and save to files
                side_1 = trimmed_coords[((orientations==1) & \
                                         np.logical_and(min(params.dist_range) <= min_dist,
                                                        min_dist <= max(params.dist_range))) ]
            
                side_0 = trimmed_coords[((orientations!=1) & \
                                         np.logical_and(min(params.dist_range) <= min_dist,
                                                        min_dist <= max(params.dist_range))) ]

                np.savetxt(f"{output_folder}/coords/{file_prefix}_side_1.txt",
                           side_1[:, restoration_order],
                           fmt="%4d")
                np.savetxt(f"{output_folder}/coords/{file_prefix}_side_0.txt",
                           side_0[:, restoration_order],
                           fmt="%4d")
                
    # Export index-file conversion table
    conversion_df = io.export_conversion_table(membrane_list=membrane_list,
                                               coords_list=coords_list)
    starfile.write(conversion_df, "./conversion_lookup.star")
