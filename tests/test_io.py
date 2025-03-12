import sys
import os
import tempfile
import unittest

import tifffile
import numpy as np

import korpuskulum
from korpuskulum import io


class IOSmokeTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tmpdir = tempfile.TemporaryDirectory()

        # Create random membrane segmentation map
        self.membrane = np.full(shape=(50, 50, 50), fill_value=1.0, dtype=int)
        self.membrane_path = f"{self.tmpdir.name}/test_mb.tiff"
        tifffile.imwrite(self.membrane_path, self.membrane, photometric="minisblack")

        # Create random particle coordinates
        self.coords_path = f"{self.tmpdir.name}/test_coords.txt"
        self.coords = np.random.randint(50, size=(10, 3), dtype=int)
        np.savetxt(self.coords_path, self.coords, fmt="%4d")

    def test_load_membrane(self):
        """
        Test the load_membrane function
        """
        segm = io.load_membrane(file_in=self.membrane_path)

        assert isinstance(
            segm, np.ndarray
        ), "Error in io.load_membrane: Output data not an ndarray."
        assert segm.shape == (
            50,
            50,
            50,
        ), "Error in io.load_membrane: Output data shape doesn't match input data shape."

    def test_load_coords(self):
        """
        Test the load_coords function
        """
        coords, n_order = io.load_coords(file_in=self.coords_path, order="zxy")
        coords_reorder, n_order_reorder = io.load_coords(
            file_in=self.coords_path, order="xyz"
        )

        assert isinstance(
            coords, np.ndarray
        ), "Error in io.load_coords: Output data not an ndarray."
        assert coords.dtype == int, "Error in io.load_coords: Output data not integers."
        assert n_order == [
            2,
            1,
            0,
        ], "Error in io.load_coords: Output numerical order wrong (should be [2,1,0])."
        assert n_order_reorder == [
            0,
            2,
            1,
        ], "Error in io.load_coords: Output numerical order wrong (should be [0,2,1])."

    @classmethod
    def tearDownClass(self):
        pass
