# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Utilities for confounds manipulation."""


def mask2vf(in_file, zooms=None, out_file=None):
    """
    Convert a binary mask on a volume fraction map.

    The algorithm simply applies a Gaussian filter with the kernel size scaled
    by the zooms given as argument.

    """
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import gaussian_filter

    img = nb.load(in_file)
    imgzooms = np.array(img.header.get_zooms()[:3], dtype=float)
    if zooms is None:
        zooms = imgzooms

    zooms = np.array(zooms, dtype=float)
    sigma = 0.5 * (zooms / imgzooms)

    data = gaussian_filter(img.get_fdata(dtype=np.float32), sigma=sigma)

    max_data = np.percentile(data[data > 0], 99)
    data = np.clip(data / max_data, a_min=0, a_max=1)

    if out_file is None:
        return data

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    nb.Nifti1Image(data.astype(np.float32), img.affine, hdr).to_filename(out_file)
    return out_file


def acompcor_masks(in_files, is_aseg=False, zooms=None):
    """
    Generate aCompCor masks.

    This function selects the CSF partial volume map from the input,
    and generates the WM and combined CSF+WM masks for aCompCor.

    The implementation deviates from Behzadi et al.
    Their original implementation thresholded the CSF and the WM partial-volume
    masks at 0.99 (i.e., 99% of the voxel volume is filled with a particular tissue),
    and then binary eroded that 2 voxels:

    > Anatomical data were segmented into gray matter, white matter,
    > and CSF partial volume maps using the FAST algorithm available
    > in the FSL software package (Smith et al., 2004). Tissue partial
    > volume maps were linearly interpolated to the resolution of the
    > functional data series using AFNI (Cox, 1996). In order to form
    > white matter ROIs, the white matter partial volume maps were
    > thresholded at a partial volume fraction of 0.99 and then eroded by
    > two voxels in each direction to further minimize partial voluming
    > with gray matter. CSF voxels were determined by first thresholding
    > the CSF partial volume maps at 0.99 and then applying a threedimensional
    > nearest neighbor criteria to minimize multiple tissue
    > partial voluming. Since CSF regions are typically small compared
    > to white matter regions mask, erosion was not applied.

    This particular procedure is not generalizable to BOLD data with different voxel zooms
    as the mathematical morphology operations will be scaled by those.
    Also, from reading the excerpt above and the tCompCor description, I (@oesteban)
    believe that they always operated slice-wise given the large slice-thickness of
    their functional data.

    Instead, *fMRIPrep*'s implementation deviates from Behzadi's implementation on two
    aspects:

      * the masks are prepared in high-resolution, anatomical space and then
        projected into BOLD space; and,
      * instead of using binary erosion, a dilated GM map is generated -- thresholding
        the corresponding PV map at 0.05 (i.e., pixels containing at least 5% of GM tissue)
        and then subtracting that map from the CSF, WM and CSF+WM (combined) masks.
        This should be equivalent to eroding the masks, except that the erosion
        only happens at direct interfaces with GM.

    When the probseg maps provene from FreeSurfer's ``recon-all`` (i.e., they are
    discrete), binary maps are *transformed* into some sort of partial volume maps
    by means of a Gaussian smoothing filter with sigma adjusted by the size of the
    BOLD data.

    """
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import binary_dilation
    from skimage.morphology import ball

    csf_file = in_files[2]  # BIDS labeling (CSF=2; last of list)
    # Load PV maps (fast) or segments (recon-all)
    gm_vf = nb.load(in_files[0])
    wm_vf = nb.load(in_files[1])
    csf_vf = nb.load(csf_file)

    # Prepare target zooms
    imgzooms = np.array(gm_vf.header.get_zooms()[:3], dtype=float)
    if zooms is None:
        zooms = imgzooms
    zooms = np.array(zooms, dtype=float)

    if not is_aseg:
        gm_data = gm_vf.get_fdata() > 0.05
        wm_data = wm_vf.get_fdata()
        csf_data = csf_vf.get_fdata()
    else:
        csf_file = mask2vf(
            csf_file,
            zooms=zooms,
            out_file=str(Path("acompcor_csf.nii.gz").absolute()),
        )
        csf_data = nb.load(csf_file).get_fdata()
        wm_data = mask2vf(in_files[1], zooms=zooms)

        # We do not have partial volume maps (recon-all route)
        gm_data = np.asanyarray(gm_vf.dataobj, np.uint8) > 0

    # Dilate the GM mask
    gm_data = binary_dilation(gm_data, structure=ball(3))

    # Output filenames
    wm_file = str(Path("acompcor_wm.nii.gz").absolute())
    combined_file = str(Path("acompcor_wmcsf.nii.gz").absolute())

    # Prepare WM mask
    wm_data[gm_data] = 0  # Make sure voxel does not contain GM
    nb.Nifti1Image(wm_data, gm_vf.affine, gm_vf.header).to_filename(wm_file)

    # Prepare combined CSF+WM mask
    comb_data = csf_data + wm_data
    comb_data[gm_data] = 0  # Make sure voxel does not contain GM
    nb.Nifti1Image(comb_data, gm_vf.affine, gm_vf.header).to_filename(combined_file)
    return [csf_file, wm_file, combined_file]
