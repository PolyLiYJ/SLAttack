# -*- coding: utf-8 -*-
"""
% Matlab script read_bntfile.m was created, Arman Savran, 2008
% Converted from the Matlab code, Kevin Mader October, 2019
% Rearranged, Arman Savran, May 2021
"""
import numpy
import numpy as np
import os
from pathlib import Path

import sys
sys.path.append("../")
np.set_printoptions(threshold=np.inf)


def fread(file_id, shape, dtype):
    """Hack np into fread from matlab"""
    count = np.prod(shape)
    return np.fromfile(file_id, dtype=dtype, count=count).reshape(shape, order='F')


def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]



def read_bntfile(filepath):
    """
    % Author: Arman Savran (arman.savran@boun.edu.tr)
    % Date:   2008
    % Outputs:
    %   zmin      : minimum depth value denoting the background
    %   nrows     : subsampled number of rows
    %   ncols     : subsampled number of columns
    %   imfile    : image file name
    %   data      : Nx5 matrix where columns are 3D coordinates and 2D
    %   normalized image coordinates respectively. 2D coordinates are
    %   normalized to the range [0,1]. N = nrows*ncols. In this matrix, values
    %   that are equal to zmin denotes the background.
    """
    with open(filepath, 'rb') as fid:
        # H is unsigned short
        nrows = fread(fid, 1, 'uint16')[0]
        ncols = fread(fid, 1, 'uint16')[0]
        zmin = fread(fid, 1, 'float64')[0]
        len1 = fread(fid, 1, 'uint16')[0]
        imfile = fread(fid, (len1, ), 'uint8')
        imfile_name = "".join(map(chr, imfile))
        assert Path(filepath).stem in imfile_name, "Names should match {} in {}".format(imfile_name, filepath.stem)
        len2 = fread(fid, 1, 'uint32')[0]
        data = fread(fid, [len2//5, 5], 'float64')
       # print(len2//5)
        data[data == zmin] = np.NAN # remove background
        data_3d = data[:,0:-2]
        mask = np.all(np.isnan(data_3d), axis=1)
        data_withoutNAN = data_3d[~mask]

       # data_withoutNAN = rand_row(data_withoutNAN, 4000)


        return nrows, ncols, data_withoutNAN

