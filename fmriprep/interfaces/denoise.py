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
"""Interfaces for thermal-noise denoising of BOLD series with ``svht_denoise``.

`svht_denoise <https://github.com/rordenlab/svht_denoise>`_ denoises a 4D series by local
PCA with the optimal singular value shrinkage of Gavish and Donoho. Given the phase, it
rotates the complex data onto the real axis before denoising, as NORDIC does.

The interface is adapted from QSIPrep's ``qsiprep.interfaces.svht.SVHTDenoise``.
"""

import os
import re
import zlib
from contextlib import contextmanager

import nibabel as nb
import numpy as np
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLine,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.mixins import reporting

LOGGER = logging.getLogger('nipype.interface')

#: Percentiles of the in-mask noise map drawn as contours in the report
NOISE_CONTOUR_PERCENTILES = (25, 50, 75)


class _SVHTDenoiseInputSpec(CommandLineInputSpec, reporting.ReportCapableInputSpec):
    in_file = File(
        exists=True,
        argstr='%s',
        position=0,
        mandatory=True,
        desc='input BOLD series (the magnitude, if phase_file is given)',
    )
    out_file = File(
        name_template='%s_denoised.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=1,
        desc='the output denoised BOLD series',
    )
    noise_image = File(
        argstr='-noise %s',
        name_template='%s_noise.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        desc='the output noise map (the noise standard deviation)',
    )
    rank_image = File(argstr='-rank %s', desc='the number of components retained per voxel')
    phase_file = File(
        exists=True,
        argstr='-phase %s',
        desc=(
            'phase of the input series. The magnitude and phase are rotated onto the '
            'real axis before denoising, and the output is then signed.'
        ),
    )
    phase_units = traits.Enum(
        'radians',
        'degrees',
        'turns',
        argstr='-phaseunits %s',
        requires=['phase_file'],
        desc=(
            'units of phase_file. Leave undefined for svht_denoise to read the unit from '
            'the observed range, which is correct for phase that spans a full turn.'
        ),
    )
    extent = traits.Int(
        argstr='-extent %d',
        desc=(
            'patch size: odd, with extent**3 greater than the number of volumes. '
            'Defaults to the smallest such value.'
        ),
    )
    filter_method = traits.Enum(
        'optshrink',
        'optthresh',
        'truncate',
        argstr='-filter %s',
        desc='how components are kept',
    )
    shape = traits.Enum('sphere', 'cube', argstr='-shape %s', desc='patch shape')
    aggregator = traits.Enum(
        'gaussian',
        'uniform',
        'exclusive',
        argstr='-aggregator %s',
        desc='how overlapping patch estimates make the output',
    )
    aggregator_fwhm = traits.Float(
        argstr='-aggregator_fwhm %g',
        desc='Gaussian aggregator width, in units of the patch-center spacing',
    )
    stride = traits.Enum(1, 2, argstr='-stride %d', desc='patch-center spacing in voxels')
    vst = traits.Bool(
        argstr='-vst %s',
        desc='variance-stabilize magnitude data before denoising',
    )
    noise_dof = traits.Range(
        low=1,
        high=64,
        argstr='-noise_dof %d',
        desc='receive channels behind each magnitude (sum of squares)',
    )
    preserve_noise_bias = traits.Bool(
        argstr='-preserve_noise_bias',
        desc='invert the variance-stabilizing transform algebraically, keeping the noise floor',
    )
    nthreads = traits.Int(argstr='-nthreads %d', nohash=True, desc='number of threads')
    out_report = File(
        'svht_denoise_report.svg',
        usedefault=True,
        hash_files=False,
        desc='filename for the visual report',
    )
    compress_report = traits.Enum(
        'auto',
        True,
        False,
        usedefault=True,
        desc=(
            "compress the reportlet with SVGO or WEBP, as niworkflows' reportlets do: "
            "'auto' compresses if the software is installed"
        ),
    )
    report_volume = traits.Range(
        low=0,
        desc=(
            'the volume the report shows. Set it to show the same volume for every echo of '
            'a run; if undefined, it is chosen from the input file name '
            '(see select_report_volume)'
        ),
    )


class _SVHTDenoiseOutputSpec(reporting.ReportCapableOutputSpec):
    out_file = File(exists=True, desc='the output denoised BOLD series')
    noise_image = File(exists=True, desc='the output noise map')
    rank_image = File(exists=True, desc='the number of components retained per voxel')


class SVHTDenoise(reporting.ReportCapableInterface, CommandLine):
    """Denoise a BOLD series by local PCA with optimal singular value shrinkage.

    The noise level is estimated with the Gavish-Donoho median estimator. Magnitude data are
    variance-stabilized first, so the Rician noise floor is removed rather than denoised.
    With ``phase_file``, the complex data are rotated onto the real axis instead, and the
    output is signed.

    The visual report shows one volume before and after denoising, with contours of the noise
    map: ``report_volume`` if set, otherwise one chosen at random but reproducibly from the
    input file's name.

    Examples
    --------
    >>> denoise = SVHTDenoise(in_file='bold.nii.gz', nthreads=2)  # doctest: +SKIP
    >>> denoise.cmdline  # doctest: +SKIP
    'svht_denoise bold.nii.gz bold_denoised.nii.gz -noise bold_noise.nii.gz -nthreads 2'
    """

    _cmd = 'svht_denoise'
    input_spec = _SVHTDenoiseInputSpec
    output_spec = _SVHTDenoiseOutputSpec

    def __init__(self, generate_report=True, **inputs):
        super().__init__(generate_report=generate_report, **inputs)

    def _format_arg(self, name, spec, value):
        if name == 'vst':
            # svht_denoise spells its switches y/n
            return spec.argstr % ('y' if value else 'n')

        if name == 'preserve_noise_bias':
            # A bare flag: present when true, absent when false
            return spec.argstr if value else ''

        return super()._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = super()._list_outputs()
        if isdefined(self.inputs.rank_image):
            outputs['rank_image'] = os.path.abspath(self.inputs.rank_image)
        return outputs

    def _generate_report(self):
        outputs = self._list_outputs()
        plot_denoising(
            in_file=self.inputs.in_file,
            denoised_file=outputs['out_file'],
            noise_file=outputs['noise_image'],
            signed=isdefined(self.inputs.phase_file),
            out_file=self._out_report,
            compress=self.inputs.compress_report,
            index=self.inputs.report_volume if isdefined(self.inputs.report_volume) else None,
        )


class _AbsoluteValueInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='a signed series')


class _AbsoluteValueOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the absolute value of the input series')


class AbsoluteValue(SimpleInterface):
    """Take the absolute value of a signed series.

    ``svht_denoise -phase`` returns the real-axis rotation of the complex data, which is
    signed. Later steps of fMRIPrep expect positive BOLD, so the magnitude is kept.
    """

    input_spec = _AbsoluteValueInputSpec
    output_spec = _AbsoluteValueOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        img = nb.load(self.inputs.in_file)
        data = np.abs(np.asanyarray(img.dataobj, dtype=np.float32))
        header = img.header.copy()
        header.set_data_dtype(np.float32)
        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix='_abs',
            newpath=runtime.cwd,
        )
        img.__class__(data, img.affine, header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def select_report_volume(in_file, n_volumes):
    """Choose the volume a denoising report shows.

    The choice is random across runs but fixed for a given file name, so that a rerun
    (or a cached node) shows the same volume. The ``echo-`` entity is ignored, so every
    echo of a run gets the same volume.

    Parameters
    ----------
    in_file : str
        The input series (any echo of the run). Only its base name is used.
    n_volumes : int
        The number of volumes in the series.

    Returns
    -------
    int
        A volume index in ``range(n_volumes)``.

    Examples
    --------
    >>> echoes = [f'sub-01_task-rest_echo-{echo}_part-mag_bold.nii.gz' for echo in (1, 2, 3)]
    >>> len({select_report_volume(echo, 200) for echo in echoes})
    1
    """
    run_name = re.sub(r'_echo-[^_]+', '', os.path.basename(str(in_file)))
    seed = zlib.crc32(run_name.encode())
    return int(np.random.default_rng(seed).integers(n_volumes))


def noise_contour_levels(noise_data, mask, percentiles=NOISE_CONTOUR_PERCENTILES):
    """Return the noise-map contour levels: percentiles of the finite, in-mask noise.

    Levels are strictly increasing, as matplotlib requires, so repeated percentiles (a flat
    noise map) collapse to fewer levels. An empty list means there is nothing to contour.
    """
    values = noise_data[mask & np.isfinite(noise_data)]
    if values.size == 0:
        return []

    levels = np.unique(np.percentile(values, percentiles))
    if levels.size == 1 and values.min() == values.max():
        # A constant map has no contour at its only value
        return []

    return [float(level) for level in levels]


@contextmanager
def _fixed_field_of_view(display):
    """Keep a nilearn display's field of view fixed while objects are added to it.

    Nilearn grows a display's axes to fit everything drawn on them, so a contour from an
    uncropped image would undo the cropping of the background. Adapted from QSIPrep.
    """
    limits = {name: (ax.ax.get_xlim(), ax.ax.get_ylim()) for name, ax in display.axes.items()}
    try:
        yield display
    finally:
        for name, ax in display.axes.items():
            xlim, ylim = limits[name]
            ax.ax.set_xlim(*xlim)
            ax.ax.set_ylim(*ylim)


def _plot_volume(
    volume_nii,
    div_id,
    label,
    cuts,
    crop_offset,
    plot_params,
    noise_nii,
    levels,
    compress='auto',
    order=('z', 'x', 'y'),
):
    """Plot one 3D volume along each cut axis, with noise-map contours."""
    from uuid import uuid4

    from lxml import etree
    from nilearn import plotting

    # compose_view needs the figure class from nireports' vendored svgutils, as nireports'
    # own reportlets (nireports.reportlets.mosaic) use
    from nireports._vendored.svgutils.transform import SVGFigure
    from nireports.reportlets.utils import SVGNS, extract_svg

    cropped_nii = volume_nii.slicer[crop_offset]
    out_files = []
    for i, mode in enumerate(order):
        params = dict(plot_params)
        params['display_mode'] = mode
        params['cut_coords'] = cuts[mode]
        params['title'] = label if i == 0 else None

        display = plotting.plot_anat(cropped_nii, **params)
        if levels:
            with _fixed_field_of_view(display):
                display.add_contours(noise_nii, levels=levels, linewidths=0.75)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Give each panel a unique id, so the two frames of the flicker can be told apart
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath(f"//{{{SVGNS}}}g[@id='figure_1']")
        find_text(xml_data)[0].set('id', f'{div_id}-{mode}-{uuid4()}')

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def plot_denoising(
    in_file,
    denoised_file,
    noise_file,
    signed,
    out_file,
    n_cuts=7,
    compress='auto',
    index=None,
):
    """Write a before/after reportlet of one volume of a denoised series.

    Parameters
    ----------
    in_file : str
        The series before denoising (the magnitude).
    denoised_file : str
        The series after denoising.
    noise_file : str
        The noise map, drawn as contours at percentiles of its in-mask values.
    signed : bool
        Whether the denoised series is signed (phase was used), in which case its absolute
        value is shown, to be comparable to the input magnitude.
    out_file : str
        The output SVG.
    n_cuts : int
        The number of slices per view.
    compress : ``'auto'`` or bool
        Whether to compress each panel with SVGO and WEBP; ``'auto'`` does so if they are
        installed.
    index : int, optional
        The volume to show. If ``None``, it is chosen with :func:`select_report_volume`.

    Returns
    -------
    str
        The absolute path of the reportlet.
    """
    from nilearn.image import crop_img, mean_img
    from nilearn.masking import compute_epi_mask
    from nireports.reportlets.utils import compose_view, cuts_from_bbox, robust_set_limits

    raw_nii = nb.load(in_file)
    denoised_nii = nb.load(denoised_file)
    n_volumes = raw_nii.shape[3] if raw_nii.ndim == 4 else 1
    if index is None:
        index = select_report_volume(in_file, n_volumes)
    elif not 0 <= index < n_volumes:
        raise ValueError(f'Report volume {index} is out of range for {n_volumes} volumes.')

    raw_vol = nb.Nifti1Image(
        np.asanyarray(raw_nii.dataobj[..., index], dtype=np.float32),
        raw_nii.affine,
    )
    denoised_data = np.asanyarray(denoised_nii.dataobj[..., index], dtype=np.float32)
    if signed:
        denoised_data = np.abs(denoised_data)

    denoised_vol = nb.Nifti1Image(denoised_data, denoised_nii.affine)

    # No brain mask exists before denoising, so derive one from the raw series
    mask_nii = compute_epi_mask(mean_img(raw_nii, copy_header=True))
    mask = np.asanyarray(mask_nii.dataobj) > 0

    noise_nii = nb.load(noise_file)
    noise_data = np.asanyarray(noise_nii.dataobj, dtype=np.float32)
    if noise_data.ndim == 4:
        noise_data = noise_data.mean(axis=3)

    noise_nii = nb.Nifti1Image(noise_data, noise_nii.affine)
    levels = noise_contour_levels(noise_data, mask)

    cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)
    _, crop_offset = crop_img(mask_nii, return_offset=True, copy_header=True)

    # The same intensity window for both frames, so they can be compared
    plot_params = robust_set_limits(raw_vol.get_fdata(dtype=np.float32)[mask], {})

    label = f'volume {index}'
    return compose_view(
        _plot_volume(
            raw_vol,
            'moving-image',
            f'Before denoising: {label}',
            cuts,
            crop_offset,
            plot_params,
            noise_nii,
            levels,
            compress=compress,
        ),
        _plot_volume(
            denoised_vol,
            'fixed-image',
            f'After denoising: {label}',
            cuts,
            crop_offset,
            plot_params,
            noise_nii,
            levels,
            compress=compress,
        ),
        out_file=out_file,
    )
