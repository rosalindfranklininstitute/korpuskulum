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

from typing import Optional

import numpy as np
import numpy.typing as npt

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def get_num_hist_bins(array_in: npt.NDArray[any],
                      *,
                      multiplier: float=2,
) -> int:
    """Calculate the number of bins for histogram generated from a given array according to the Rice Rule. Multiplier of Rice Rule can be changed.

    Args:
    array_in (ndarray)           : Array for generating histogram
    multiplier (optional, float) : Multiplier of Rice Rule. Default = 2

    Returns:
    int
    """
    nbins = multiplier * np.ceil(np.cbrt(len(array_in))).astype(int)
    
    return nbins


def plot_polar_hist(dist_array: npt.NDArray[any],
                    angle_array: npt.NDArray[any],
                    *,
                    dist_cutoff: Optional[float]=None,
                    colormap: str="gist_heat_r",
                    savefig: Optional[str]=None,
):
    """Plot, on screen or save as file, the polar histogram of particle distribution given arrays including information about particle-membrane distances and angles.

    Args:
    dist_array (ndarray) : Array containing particle-membrane minimum distances
    angle_array (ndarray) : Array containing particle-membrane angles
    dist_cutoff (optional, float) : Cutoff distance for polar histogram plotting. Default = None
    colormap (optional, str) : Matplotlib colormap for histogram display. Default = gist_heat_r
    savefig (optional, str) : Path to polar histogram figure being saved if value provided. Default = None

    Returns:
    None
    """
    # Set cutoff distance to maximum particle distance if unspecified
    if dist_cutoff is None:
        dist_cutoff = dist_array.max()

    criteria = np.logical_and(
        0.1 < dist_array,
        dist_array <= dist_cutoff
    )

    # Estimate optimal number of bins for radial and angular components
    rbins = np.linspace(0, dist_cutoff, get_num_hist_bins(dist_array[criteria]))
    abins = np.linspace(-np.pi, np.pi, get_num_hist_bins(angle_array[criteria]))

    # Produce histogram and save (display)
    hist, _, _ = np.histogram2d(angle_array[criteria],
                                dist_array[criteria],
                                bins=(abins, rbins))
    A, R = np.meshgrid(abins, rbins)

    fig, ax = plt.subplots(figsize=(8,6),
                           subplot_kw=dict(projection="polar")
    )
    pc = ax.pcolormesh(A, R, hist.T,
                       cmap=colormap, vmax=hist.max()
    )
    fig.colorbar(pc)
    plt.tight_layout()

    if savefig is not None:
        fig.savefig(savefig)
        plt.close(fig)


def plot_min_dist_hist(dist_array: npt.NDArray[any],
                       orientations: npt.NDArray[any],
                       protein_name: str,
                       membrane_name: str,
                       *,
                       dist_low: Optional[float]=2,
                       dist_high: Optional[float]=10,
                       savefig: Optional[str]=None,
):
    """Plot, on screen or save as file, the histogram of minimum particle-membrane distances, separated by their locations (side I vs side O).

    Args:
    dist_array (ndarray)        : Array containing particle-membrane minimum distances
    orientations (ndarray)      : Array containing side tags of particles
    protein_name (str)          : Name of protein of interest
    membrane_name (str)         : Name of membrane of interest
    dist_low (optional, float)  : Minimum distance for histogram plotting. Default = 2
    dist_high (optional, float) : Maximum distance for histogram plotting. Default = 10
    savefig (optional, str)     : Path to polar histogram figure being saved if value provided. Default = None

    Returns:
    None
    """
    crit_1 = (orientations == 1)
    crit_2 = np.logical_and(
        dist_low <= dist_array,
        dist_array <= dist_high,
    )

    fig, ax = plt.subplots()
    hist_side_1 = plt.hist(
        dist_array[(crit_1 & crit_2)],
        get_num_hist_bins(dist_array[(crit_1 & crit_2)]),
        alpha=0.75,
        label=f"{protein_name}, Membrane {membrane_name}, Side I"
    )
    hist_side_0 = plt.hist(
        dist_array[(~crit_1 & crit_2)],
        get_num_hist_bins(dist_array[(~crit_1 & crit_2)]),
        alpha=0.75,
        label=f"{protein_name}, Membrane {membrane_name}, Side O"
    )
    ax.legend()

    if savefig is not None:
        fig.savefig(savefig)
        plt.close()
