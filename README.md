# MonaiMatrixTransforms
Expands the Monai transform library so that a random affine matrix can be created and applied with one resampling.

## Motivation
Many of the transformations usually applied to an image are linear, such as translation, flip, zoom, and shear. This means that they do not need to be applied in sequence but can be applied in one resampling step. This is not taken advantage of in the original Monai implementation. This library offers the functionality to take advantage of the linearity of the transformations. 

## Usage
In this library the transformations are first applied to a homogenious transformation matrix, and when all transformations are done, the matrix is used to resample the image. This library only supports the dictionary version of Transforms.

### Example
```
Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=['image']),
        MatInitializer(keys=['aff_mat'], dim=3), # Initialize the homogeneous transformation matrix
        MatRandSheard(keys=['aff_mat'], range=(0.1, 0.1, 0.1)),
        MatRandZoomd(keys=['aff_mat'], range=(0.5, 2.0)),
        MatRandRotationd(keys=['aff_mat'], range=(0.1, 0.1, 0.1)),
        MatRandTranslated(keys=['aff_mat'], range=(20, 20, 20)),
        ApplyAffined(aff_key='aff_mat', img_key='image', spatial_size=img_size, 
            mode='nearest', padding_mode='zeros'), # Apply the homogeneous transformation matrix to the image
    ]
)
```
