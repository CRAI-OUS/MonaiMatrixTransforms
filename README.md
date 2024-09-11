# MonaiMatrixTransforms

MonaiMatrixTransforms extends the MONAI transformation library, enabling the application of random affine transformations using a single resampling step, reducing the computational cost compared to performing multiple resamplings for each individual transformation.

## Motivation
In medical imaging, many commonly applied transformations such as translation, flipping, zoom, and shear are linear. This allows them to be combined into a single affine transformation, reducing the need for sequential resampling steps that can be computationally expensive. However, in the original MONAI library, these transformations are often applied separately, missing the opportunity for optimization.

MonaiMatrixTransforms addresses this by consolidating these transformations into a homogeneous affine matrix, which can be applied to the image in one resampling operation, improving efficiency without compromising accuracy.

## Usage
In this library, transformations are applied to a homogeneous transformation matrix. Once all desired transformations are incorporated, the matrix is then used to resample the image in a single step.
```
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monaimatrixtransforms import (
    MatInitializer,
    MatRandSheard,
    MatRandZoomd,
    MatRandRotationd,
    MatRandTranslated,
    ApplyAffined
)

# Define your image size
img_size = (128, 128, 128)  # Example image size

# Compose the transformations
transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=['image']),
        MatInitializer(keys=['aff_mat'], dim=3),  # Initialize the homogeneous transformation matrix
        MatRandSheard(keys=['aff_mat'], range=(0.1, 0.1, 0.1)),
        MatRandZoomd(keys=['aff_mat'], range=(0.5, 2.0)),
        MatRandRotationd(keys=['aff_mat'], range=(0.1, 0.1, 0.1)),
        MatRandTranslated(keys=['aff_mat'], range=(20, 20, 20)),
        ApplyAffined(
            aff_key='aff_mat', 
            img_key='image', 
            spatial_size=img_size, 
            mode='nearest', 
            padding_mode='zeros'  # Apply the affine matrix to the image
        ),
    ]
)
```
