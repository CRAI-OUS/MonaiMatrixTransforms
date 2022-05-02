"""
This file contains an extension of  monais.transforms that allows for doing multiple 
affine transforms on a single image with only a single call to the resampler.

Example:
Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=['image']),
        MatInitializer(keys=['aff_mat'], dim=3),
        MatRandSheard(keys=['aff_mat'], range=(0.1, 0.1, 0.1)),
        MatRandZoomd(keys=['aff_mat'], range=(0.5, 2.0)),
        MatRandRotationd(keys=['aff_mat'], range=(0.1, 0.1, 0.1)),
        MatRandTranslated(keys=['aff_mat'], range=(20, 20, 20)),
        ApplyAffined(aff_key='aff_mat', img_key='image', spatial_size=img_size, 
            mode='nearest', padding_mode='zeros'),
    ]
)
"""
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.spatial.array import (
    AffineGrid,
    Resample,
)
from monai.utils.module import look_up_option
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.transform import Randomizable, RandomizableTransform, ThreadUnsafe, Transform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    fall_back_tuple,
)

class MatInitializer(MapTransform):
    """
    Initialize an identity matrix.

    Args:
        keys: Keys for each affine matrix to be initialized.
        dim: Dimension of the image to be transformed. Can be 2 or 3.
    """
    
    def __init__(self, keys: KeysCollection, dim: int) -> None:
        super().__init__(keys=keys, allow_missing_keys=True)
        self.dim = dim
        self.mat = np.eye(dim+1)
        self.keys = keys

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.mat
        return d


class ApplyAffined(MapTransform):
    """
    Apply affine transform to the image from a affine matrix.
    Args:
        aff_key: Key for the affine matrix.
        img_key: Key for the image.
        spatial_size: Size of the output image.
        mode: Interpolation mode.
        padding_mode: Padding mode.
        device: Device on which the tensors will be allocated.
    """

    backend = list(set(AffineGrid.backend) & set(Resample.backend))

    def __init__(
        self,
        aff_key: Hashable,
        img_key: Hashable,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
        image_only: bool = False,
    ) -> None:
        """
        Applies affine transform to the image.
        Args:
            aff_key: Key for the affine matrix.
            img_key: Key for the image.
            spatial_size: Size of the output image.
            mode: Interpolation mode.
            padding_mode: Padding mode.
            device: Device on which the tensors will be allocated.

        """
        super().__init__(keys=[aff_key, img_key], allow_missing_keys=True)
        self.aff_key = aff_key
        self.img_key = img_key
        self.resampler = Resample(device=device)
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        affine_grid = AffineGrid(
            affine=d[self.aff_key],
            device=self.resampler.device,
        )
        img = d[self.img_key]
        sp_size = fall_back_tuple(self.spatial_size, img.shape[1:])
        grid, affine = affine_grid(spatial_size=sp_size)
        ret = self.resampler(img, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        d[self.img_key] = ret
        return d

class MatRandTranslated(MapTransform):
    """
    Applies a random translation to an affine matrix.
    Args:
        keys: Keys to pick data for transformation.
        range: Range of translation.
        allow_missing_keys: don't raise exception if key is missing.
    """
    def __init__(
        self,
        keys: KeysCollection,
        range: Optional[Union[Sequence[float], float]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.range = range

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            dim = d[key].shape[-1]-1 # Dimension of the image to be transformed.
            aff_mat = np.eye(dim + 1)
            if isinstance(self.range, (tuple, list)):
                if len(self.range) != dim:
                    raise ValueError(f"range should be a tuple or list of {dim} elements.")
                for i in range(dim):
                    aff_mat[i, -1] = np.random.uniform(-self.range[i], self.range[i])     
            elif isinstance(self.range, float):
                aff_mat[:-1, -1] = np.random.uniform(-self.range, self.range, size=dim)
            d[key] = aff_mat@d[key]
        return d

class MatRandZoomd(MapTransform):
    """
    Applies a random isotropic zoom to an affine matrix.
    Args:
        keys: Keys to pick data for transformation.
        range: Tupel with range of zoom. 0.5 is half zoom. 2 is double zoom.
        allow_missing_keys: don't raise exception if key is missing.
    """
    def __init__(
        self,
        keys: KeysCollection,
        range: Optional[Union[Sequence[float], float]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.range = range

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            dim = d[key].shape[-1]-1 # Dimension of the image to be transformed.
            aff_mat = np.eye(dim + 1)
            if isinstance(self.range, (tuple, list)):
                if len(self.range) != 2:
                    raise ValueError(f"range should be a tuple or list of 2 elements.")
                zoom_coef = np.random.uniform(self.range[0], self.range[1])
                for i in range(dim):
                    aff_mat[i, i] = 1/zoom_coef
            elif isinstance(self.range, float):
                raise ValueError("range should be a tuple or list of 2 elements.")
            d[key] = aff_mat@d[key]
        return d


class MatRandScale(MapTransform):
    """
    Applies a random scale to an affine matrix.
    Args:
        keys: Keys to pick data for transformation.
        range: Range of scale. Must be a tuple or list with length equal to de dimension of the affine matrix. Each element must contain a tuple of 2 elements. 
        A value of 0.5 means that the image will be scaled to half its original size and a value of 2 means that the image will be scaled to twice its original size. 
        allow_missing_keys: don't raise exception if key is missing.
    """
    def __init__(
        self,
        keys: KeysCollection,
        range: Optional[Union[Sequence[float], float]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.range = range

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            dim = d[key].shape[-1]-1 # Dimension of the image to be transformed.
            aff_mat = np.eye(dim + 1)
            if isinstance(self.range, (tuple, list)) and len(self.range) == dim:
                for i in range(dim):
                    if len(self.range[i]) != 2:
                        raise ValueError(f"range should be a tuple or list of 2 elements.")
                    aff_mat[i, i] = 1/np.random.uniform(self.range[i][0], self.range[i][1])
            else:
                raise ValueError(f"range should be a tuple or list of {dim} elements.")
            d[key] = aff_mat@d[key]
        return d

class MatRandSheard(MapTransform):
    """
    Applies a random sheard to an affine matrix.
    Args:
        keys: Keys to pick data for transformation.
        range: Range of sheard. Must be a tuple or list with length equal to de dimension of the affine matrix. Each element must contain a float.
    """
    def __init__(
        self,
        keys: KeysCollection,
        range: Optional[Union[Sequence[float], float]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.range = range

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            dim = d[key].shape[-1]-1
            aff_mat = np.eye(dim + 1)
            if isinstance(self.range, (tuple, list)) and len(self.range) == dim:
                if dim == 2:
                    aff_mat[0, 1] = np.random.uniform(-self.range[0], self.range[0])
                elif dim == 3:
                    aff_mat[0, 1] = np.random.uniform(-self.range[0], self.range[0])
                    aff_mat[0, 2] = np.random.uniform(-self.range[1], self.range[1])
                    aff_mat[1, 2] = np.random.uniform(-self.range[2], self.range[2])
            else:
                raise ValueError(f"range should be a tuple or list of {dim} elements.")
            d[key] = aff_mat@d[key]
        return d


class MatRandRotationd(MapTransform):
    """
    Applies a random rotation to an affine matrix.
    Args:
        keys: Keys to pick data for transformation.
        range: If range is None a uniformly sampled random rotation is generated. 
            If range is a float or a list of floats, a random euler rotation around xyz is generated.
            Mark that this orientation is not uniformly sampled.
        allow_missing_keys: don't raise exception if key is missing.
    """
    def __init__(
        self,
        keys: KeysCollection,
        range: Optional[Union[Sequence[float], float]] = None,
        img_key: KeysCollection = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.range = range
        self.img_key = img_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            dim = d[key].shape[-1]-1 # Dimension of the image to be transformed.

            if self.range is None:
                rot_mat = self._uniform_random_rotation_mat(dim)
            else:
                rot_mat = self._euler_random_rotation_mat(dim, self.range)
            rot_aff_mat = np.eye(dim+1)
            rot_aff_mat[:-1, :-1] = rot_mat

            d[key] = rot_aff_mat@d[key]
        return d
    def _rotation_mat(self, dim:int, angle:float, axis:int=0):
        if dim == 2:
            return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        elif dim == 3:
            if axis == 0:
                return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            elif axis == 1:
                return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            elif axis == 2:
                return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            else:
                raise ValueError(f"axis should be in [0, 1, 2].")

    def _euler_random_rotation_mat(self, dim:int, range:Sequence[float]):
        if dim == 2:
            theta = np.random.uniform(-range[0], range[0])
            return self._rotation_mat(dim, theta)
        elif dim == 3:
            if isinstance(range, float):
                theta = np.random.uniform(-range, range)
                phi = np.random.uniform(-range, range)
                psi = np.random.uniform(-range, range)
            elif len(range) == 3:
                theta = np.random.uniform(-range[0], range[0])
                phi = np.random.uniform(-range[1], range[1])
                psi = np.random.uniform(-range[2], range[2]) 
            else:
                raise ValueError(f"range should be a tuple or list of 3 elements or a float.")
            rot1 = self._rotation_mat(dim, theta, axis=0)
            rot2 = self._rotation_mat(dim, phi, axis=1)
            rot3 = self._rotation_mat(dim, psi, axis=2)
            
            return rot1@rot2@rot3
            
    def _uniform_random_rotation_mat(self, dim: int):
        """
        Taken from: https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/
        Apply a random rotation matrix in 3D, with a distribution uniform over the
        sphere.
        Arguments:
            x: vector or set of vectors with dimension (n, 3), where n is the
                number of vectors
        Returns:
            Array of shape (n, 3) containing the randomly rotated vectors of x,
            about the mean coordinate of x.
        Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
        https://doi.org/10.1016/B978-0-08-050755-2.50034-8
        """

        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = -R[0, 1]

        if dim==2:
            return R

        # There are two random variables in [0, 1) here (naming is same as paper)
        x2 = 2 * np.pi * np.random.rand()
        x3 = np.random.rand()
        # Rotation of all points around x axis using matrix
        v = np.array([
            np.cos(x2) * np.sqrt(x3),
            np.sin(x2) * np.sqrt(x3),
            np.sqrt(1 - x3)
        ])
        H = np.eye(3) - (2 * np.outer(v, v))
        M = -(H @ R)
        return M