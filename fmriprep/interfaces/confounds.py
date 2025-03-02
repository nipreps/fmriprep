# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""
Handling confounds.

    .. testsetup::

    >>> import os
    >>> import pandas as pd

"""

import os
import re

import nibabel as nb
import nitransforms as nt
import numpy as np
import pandas as pd
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from nireports.reportlets.modality.func import fMRIPlot
from niworkflows.utils.timeseries import _cifti_timeseries, _nifti_timeseries
from scipy import ndimage as ndi
from scipy.spatial import transform as sst

LOGGER = logging.getLogger('nipype.interface')


class _aCompCorMasksInputSpec(BaseInterfaceInputSpec):
    in_vfs = InputMultiObject(File(exists=True), desc='Input volume fractions.')
    is_aseg = traits.Bool(
        False, usedefault=True, desc="Whether the input volume fractions come from FS' aseg."
    )
    bold_zooms = traits.Tuple(
        traits.Float, traits.Float, traits.Float, mandatory=True, desc='BOLD series zooms'
    )


class _aCompCorMasksOutputSpec(TraitedSpec):
    out_masks = OutputMultiObject(
        File(exists=True), desc='CSF, WM and combined masks, respectively'
    )


class aCompCorMasks(SimpleInterface):
    """Generate masks in T1w space for aCompCor."""

    input_spec = _aCompCorMasksInputSpec
    output_spec = _aCompCorMasksOutputSpec

    def _run_interface(self, runtime):
        from ..utils.confounds import acompcor_masks

        self._results['out_masks'] = acompcor_masks(
            self.inputs.in_vfs,
            self.inputs.is_aseg,
            self.inputs.bold_zooms,
        )
        return runtime


class _FSLRMSDeviationInputSpec(BaseInterfaceInputSpec):
    xfm_file = File(exists=True, mandatory=True, desc='Head motion transform file')
    boldref_file = File(exists=True, mandatory=True, desc='BOLD reference file')


class _FSLRMSDeviationOutputSpec(TraitedSpec):
    out_file = File(desc='Output motion parameters file')


class FSLRMSDeviation(SimpleInterface):
    """Reconstruct FSL root mean square deviation from affine transforms."""

    input_spec = _FSLRMSDeviationInputSpec
    output_spec = _FSLRMSDeviationOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.boldref_file, suffix='_motion.tsv', newpath=runtime.cwd
        )

        boldref = nb.load(self.inputs.boldref_file)
        hmc = nt.linear.load(self.inputs.xfm_file)

        center = 0.5 * (np.array(boldref.shape[:3]) - 1) * boldref.header.get_zooms()[:3]

        # Revert to vox2vox transforms
        fsl_hmc = nt.io.fsl.FSLLinearTransformArray.from_ras(
            hmc.matrix, reference=boldref, moving=boldref
        )
        fsl_matrix = np.stack([xfm['parameters'] for xfm in fsl_hmc.xforms])

        diff = fsl_matrix[1:] @ np.linalg.inv(fsl_matrix[:-1]) - np.eye(4)
        M = diff[:, :3, :3]
        t = diff[:, :3, 3] + M @ center
        Rmax = 80.0

        rmsd = np.concatenate(
            [
                [np.nan],
                np.sqrt(
                    np.diag(t @ t.T)
                    + np.trace(M.transpose(0, 2, 1) @ M, axis1=1, axis2=2) * Rmax**2 / 5
                ),
            ]
        )

        params = pd.DataFrame(data=rmsd, columns=['rmsd'])
        params.to_csv(self._results['out_file'], sep='\t', index=False, na_rep='n/a')

        return runtime


class _FSLMotionParamsInputSpec(BaseInterfaceInputSpec):
    xfm_file = File(exists=True, desc='Head motion transform file')
    boldref_file = File(exists=True, desc='BOLD reference file')


class _FSLMotionParamsOutputSpec(TraitedSpec):
    out_file = File(desc='Output motion parameters file')


class FSLMotionParams(SimpleInterface):
    """Reconstruct FSL motion parameters from affine transforms."""

    input_spec = _FSLMotionParamsInputSpec
    output_spec = _FSLMotionParamsOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.boldref_file, suffix='_motion.tsv', newpath=runtime.cwd
        )

        boldref = nb.load(self.inputs.boldref_file)
        hmc = nt.linear.load(self.inputs.xfm_file)

        # FSL's "center of gravity" is the center of mass scaled by zooms
        # No rotation is applied.
        center_of_gravity = np.matmul(
            np.diag(boldref.header.get_zooms()),
            ndi.center_of_mass(np.asanyarray(boldref.dataobj)),
        )

        # Revert to vox2vox transforms
        fsl_hmc = nt.io.fsl.FSLLinearTransformArray.from_ras(
            hmc.matrix, reference=boldref, moving=boldref
        )
        fsl_matrix = np.stack([xfm['parameters'] for xfm in fsl_hmc.xforms])

        # FSL uses left-handed rotation conventions, so transpose
        mats = fsl_matrix[:, :3, :3].transpose(0, 2, 1)

        # Rotations are recovered directly
        rot_xyz = sst.Rotation.from_matrix(mats).as_euler('XYZ')
        # Translations are recovered by applying the rotation to the center of gravity
        trans_xyz = fsl_matrix[:, :3, 3] - mats @ center_of_gravity + center_of_gravity

        params = pd.DataFrame(
            data=np.hstack((trans_xyz, rot_xyz)),
            columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
        )

        params.to_csv(self._results['out_file'], sep='\t', index=False, na_rep='n/a')

        return runtime


class _FramewiseDisplacementInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='Head motion transform file')
    radius = traits.Float(50, usedefault=True, desc='Radius of the head in mm')


class _FramewiseDisplacementOutputSpec(TraitedSpec):
    out_file = File(desc='Output framewise displacement file')


class FramewiseDisplacement(SimpleInterface):
    """Calculate framewise displacement estimates."""

    input_spec = _FramewiseDisplacementInputSpec
    output_spec = _FramewiseDisplacementOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_fd.tsv', newpath=runtime.cwd
        )

        motion = pd.read_csv(self.inputs.in_file, delimiter='\t')

        # Filter and ensure we have all parameters
        diff = motion[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].diff()
        diff[['rot_x', 'rot_y', 'rot_z']] *= self.inputs.radius

        fd = pd.DataFrame(diff.abs().sum(axis=1, skipna=False), columns=['FramewiseDisplacement'])

        fd.to_csv(self._results['out_file'], sep='\t', index=False, na_rep='n/a')

        return runtime


class _FilterDroppedInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input CompCor metadata')


class _FilterDroppedOutputSpec(TraitedSpec):
    out_file = File(desc='filtered CompCor metadata')


class FilterDropped(SimpleInterface):
    """Filter dropped components from CompCor metadata files

    Uses the boolean ``retained`` column to identify rows to keep or filter.
    """

    input_spec = _FilterDroppedInputSpec
    output_spec = _FilterDroppedOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_filtered', use_ext=True, newpath=runtime.cwd
        )

        metadata = pd.read_csv(self.inputs.in_file, sep='\t')
        metadata[metadata.retained].to_csv(
            self._results['out_file'], sep='\t', index=False, na_rep='n/a'
        )

        return runtime


class _RenameACompCorInputSpec(BaseInterfaceInputSpec):
    components_file = File(exists=True, desc='input aCompCor components')
    metadata_file = File(exists=True, desc='input aCompCor metadata')


class _RenameACompCorOutputSpec(TraitedSpec):
    components_file = File(desc='output aCompCor components')
    metadata_file = File(desc='output aCompCor metadata')


class RenameACompCor(SimpleInterface):
    """Rename ACompCor components based on their masks

    Components from the "CSF" mask are ``c_comp_cor_*``.
    Components from the "WM" mask are ``w_comp_cor_*``.
    Components from the "combined" mask are ``a_comp_cor_*``.

    Each set of components is renumbered to start at ``?_comp_cor_00``.
    """

    input_spec = _RenameACompCorInputSpec
    output_spec = _RenameACompCorOutputSpec

    def _run_interface(self, runtime):
        try:
            components = pd.read_csv(self.inputs.components_file, sep='\t')
            metadata = pd.read_csv(self.inputs.metadata_file, sep='\t')
        except pd.errors.EmptyDataError:
            # Can occur when testing on short datasets; otherwise rare
            self._results['components_file'] = self.inputs.components_file
            self._results['metadata_file'] = self.inputs.metadata_file
            return runtime

        self._results['components_file'] = fname_presuffix(
            self.inputs.components_file, suffix='_renamed', use_ext=True, newpath=runtime.cwd
        )
        self._results['metadata_file'] = fname_presuffix(
            self.inputs.metadata_file, suffix='_renamed', use_ext=True, newpath=runtime.cwd
        )

        all_comp_cor = metadata[metadata['retained']]

        c_comp_cor = all_comp_cor[all_comp_cor['mask'] == 'CSF']
        w_comp_cor = all_comp_cor[all_comp_cor['mask'] == 'WM']
        a_comp_cor = all_comp_cor[all_comp_cor['mask'] == 'combined']

        c_orig = c_comp_cor['component']
        c_new = [f'c_comp_cor_{i:02d}' for i in range(len(c_orig))]

        w_orig = w_comp_cor['component']
        w_new = [f'w_comp_cor_{i:02d}' for i in range(len(w_orig))]

        a_orig = a_comp_cor['component']
        a_new = [f'a_comp_cor_{i:02d}' for i in range(len(a_orig))]

        final_components = components.rename(columns=dict(zip(c_orig, c_new, strict=False)))
        final_components.rename(columns=dict(zip(w_orig, w_new, strict=False)), inplace=True)
        final_components.rename(columns=dict(zip(a_orig, a_new, strict=False)), inplace=True)
        final_components.to_csv(
            self._results['components_file'], sep='\t', index=False, na_rep='n/a'
        )

        metadata.loc[c_comp_cor.index, 'component'] = c_new
        metadata.loc[w_comp_cor.index, 'component'] = w_new
        metadata.loc[a_comp_cor.index, 'component'] = a_new

        metadata.to_csv(self._results['metadata_file'], sep='\t', index=False, na_rep='n/a')

        return runtime


class GatherConfoundsInputSpec(BaseInterfaceInputSpec):
    signals = File(exists=True, desc='input signals')
    dvars = File(exists=True, desc='file containing DVARS')
    std_dvars = File(exists=True, desc='file containing standardized DVARS')
    fd = File(exists=True, desc='input framewise displacement')
    rmsd = File(exists=True, desc='input RMS framewise displacement')
    tcompcor = File(exists=True, desc='input tCompCorr')
    acompcor = File(exists=True, desc='input aCompCorr')
    crowncompcor = File(exists=True, desc='input crown-based regressors')
    cos_basis = File(exists=True, desc='input cosine basis')
    motion = File(exists=True, desc='input motion parameters')


class GatherConfoundsOutputSpec(TraitedSpec):
    confounds_file = File(exists=True, desc='output confounds file')
    confounds_list = traits.List(traits.Str, desc='list of headers')


class GatherConfounds(SimpleInterface):
    r"""
    Combine various sources of confounds in one TSV file

    >>> pd.DataFrame({'a': [0.1]}).to_csv('signals.tsv', index=False, na_rep='n/a')
    >>> pd.DataFrame({'b': [0.2]}).to_csv('dvars.tsv', index=False, na_rep='n/a')

    >>> gather = GatherConfounds()
    >>> gather.inputs.signals = 'signals.tsv'
    >>> gather.inputs.dvars = 'dvars.tsv'
    >>> res = gather.run()
    >>> res.outputs.confounds_list
    ['Global signals', 'DVARS']

    >>> pd.read_csv(res.outputs.confounds_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
         a    b
    0  0.1  0.2

    """

    input_spec = GatherConfoundsInputSpec
    output_spec = GatherConfoundsOutputSpec

    def _run_interface(self, runtime):
        combined_out, confounds_list = _gather_confounds(
            signals=self.inputs.signals,
            dvars=self.inputs.dvars,
            std_dvars=self.inputs.std_dvars,
            fdisp=self.inputs.fd,
            rmsd=self.inputs.rmsd,
            tcompcor=self.inputs.tcompcor,
            acompcor=self.inputs.acompcor,
            crowncompcor=self.inputs.crowncompcor,
            cos_basis=self.inputs.cos_basis,
            motion=self.inputs.motion,
            newpath=runtime.cwd,
        )
        self._results['confounds_file'] = combined_out
        self._results['confounds_list'] = confounds_list
        return runtime


def _gather_confounds(
    signals=None,
    dvars=None,
    std_dvars=None,
    fdisp=None,
    rmsd=None,
    tcompcor=None,
    acompcor=None,
    crowncompcor=None,
    cos_basis=None,
    motion=None,
    newpath=None,
):
    r"""
    Load confounds from the filenames, concatenate together horizontally
    and save new file.

    >>> pd.DataFrame({'Global Signal': [0.1]}).to_csv('signals.tsv', index=False, na_rep='n/a')
    >>> pd.DataFrame({'stdDVARS': [0.2]}).to_csv('dvars.tsv', index=False, na_rep='n/a')
    >>> out_file, confound_list = _gather_confounds('signals.tsv', 'dvars.tsv')
    >>> confound_list
    ['Global signals', 'DVARS']

    >>> pd.read_csv(out_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
       global_signal  std_dvars
    0            0.1        0.2


    """

    def less_breakable(a_string):
        """hardens the string to different envs (i.e., case insensitive, no whitespace, '#'"""
        return ''.join(a_string.split()).strip('#')

    # Taken from https://stackoverflow.com/questions/1175208/
    # If we end up using it more than just here, probably worth pulling in a well-tested package
    def camel_to_snake(name):
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _adjust_indices(left_df, right_df):
        # This forces missing values to appear at the beginning of the DataFrame
        # instead of the end
        index_diff = len(left_df.index) - len(right_df.index)
        if index_diff > 0:
            right_df.index = range(index_diff, len(right_df.index) + index_diff)
        elif index_diff < 0:
            left_df.index = range(-index_diff, len(left_df.index) - index_diff)

    all_files = []
    confounds_list = []
    for confound, name in (
        (signals, 'Global signals'),
        (std_dvars, 'Standardized DVARS'),
        (dvars, 'DVARS'),
        (fdisp, 'Framewise displacement'),
        (rmsd, 'Framewise displacement (RMS)'),
        (tcompcor, 'tCompCor'),
        (acompcor, 'aCompCor'),
        (crowncompcor, 'crownCompCor'),
        (cos_basis, 'Cosine basis'),
        (motion, 'Motion parameters'),
    ):
        if confound is not None and isdefined(confound):
            confounds_list.append(name)
            if os.path.exists(confound) and os.stat(confound).st_size > 0:
                all_files.append(confound)

    confounds_data = pd.DataFrame()
    for file_name in all_files:  # assumes they all have headings already
        try:
            new = pd.read_csv(file_name, sep='\t')
        except pd.errors.EmptyDataError:
            # No data, nothing to concat
            continue
        for column_name in new.columns:
            new.rename(
                columns={column_name: camel_to_snake(less_breakable(column_name))}, inplace=True
            )

        _adjust_indices(confounds_data, new)
        confounds_data = pd.concat((confounds_data, new), axis=1)

    if newpath is None:
        newpath = os.getcwd()

    combined_out = os.path.join(newpath, 'confounds.tsv')
    confounds_data.to_csv(combined_out, sep='\t', index=False, na_rep='n/a')

    return combined_out, confounds_list


class _FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_nifti = File(exists=True, mandatory=True, desc='input BOLD (4D NIfTI file)')
    in_cifti = File(exists=True, desc='input BOLD (CIFTI dense timeseries)')
    in_segm = File(exists=True, desc='volumetric segmentation corresponding to in_nifti')
    confounds_file = File(exists=True, desc="BIDS' _confounds.tsv file")

    str_or_tuple = traits.Either(
        traits.Str,
        traits.Tuple(traits.Str, traits.Either(None, traits.Str)),
        traits.Tuple(traits.Str, traits.Either(None, traits.Str), traits.Either(None, traits.Str)),
    )
    confounds_list = traits.List(
        str_or_tuple, minlen=1, desc='list of headers to extract from the confounds_file'
    )
    tr = traits.Either(None, traits.Float, usedefault=True, desc='the repetition time')
    drop_trs = traits.Int(0, usedefault=True, desc='dummy scans')


class _FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class FMRISummary(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """

    input_spec = _FMRISummaryInputSpec
    output_spec = _FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_nifti, suffix='_fmriplot.svg', use_ext=False, newpath=runtime.cwd
        )

        has_cifti = isdefined(self.inputs.in_cifti)

        # Read input object and create timeseries + segments object
        seg_file = self.inputs.in_segm if isdefined(self.inputs.in_segm) else None
        dataset, segments = _nifti_timeseries(
            nb.load(self.inputs.in_nifti),
            nb.load(seg_file),
            remap_rois=False,
            labels=(
                ('WM+CSF', 'Edge')
                if has_cifti
                else ('Ctx GM', 'dGM', 'sWM+sCSF', 'dWM+dCSF', 'Cb', 'Edge')
            ),
        )

        # Process CIFTI
        if has_cifti:
            cifti_data, cifti_segments = _cifti_timeseries(nb.load(self.inputs.in_cifti))

            if seg_file is not None:
                # Append WM+CSF and Edge masks
                cifti_length = cifti_data.shape[0]
                dataset = np.vstack((cifti_data, dataset))
                segments = {k: np.array(v) + cifti_length for k, v in segments.items()}
                cifti_segments.update(segments)
                segments = cifti_segments
            else:
                dataset, segments = cifti_data, cifti_segments

        dataframe = pd.read_csv(
            self.inputs.confounds_file,
            sep='\t',
            index_col=None,
            dtype='float32',
            na_filter=True,
            na_values='n/a',
        )

        headers = []
        units = {}
        names = {}

        for conf_el in self.inputs.confounds_list:
            if isinstance(conf_el, list | tuple):
                headers.append(conf_el[0])
                if conf_el[1] is not None:
                    units[conf_el[0]] = conf_el[1]

                if len(conf_el) > 2 and conf_el[2] is not None:
                    names[conf_el[0]] = conf_el[2]
            else:
                headers.append(conf_el)

        if not headers:
            data = None
            units = None
        else:
            data = dataframe[headers]

        data = data.rename(columns=names)

        fig = fMRIPlot(
            dataset,
            segments=segments,
            tr=self.inputs.tr,
            confounds=data,
            units=units,
            nskip=self.inputs.drop_trs,
            paired_carpet=has_cifti,
        ).plot()
        fig.savefig(self._results['out_file'], bbox_inches='tight')
        return runtime
