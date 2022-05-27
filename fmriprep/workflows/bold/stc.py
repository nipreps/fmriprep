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
"""
Slice-Timing Correction (STC) of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_stc_wf

"""
import numpy as np
import nibabel as nb
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, afni
from nipype.interfaces.base import isdefined

from ... import config


LOGGER = config.loggers.workflow


class TShift(afni.TShift):
    """Patched version of TShift implementing the "TooShort" behavior."""

    def _pre_run_hook(self, runtime):
        ignore = self.inputs.ignore if isdefined(self.inputs.ignore) else 0
        ntsteps = nb.load(self.inputs.in_file).shape[3]
        if ntsteps - ignore < 5:
            raise RuntimeError(
                f"Insufficient length of BOLD data ({ntsteps} time points) after "
                f"discarding {ignore} nonsteady-state (or 'dummy') time points."
            )
        return runtime


def init_bold_stc_wf(metadata, name='bold_stc_wf'):
    """
    Create a workflow for :abbr:`STC (slice-timing correction)`.

    This workflow performs :abbr:`STC (slice-timing correction)` over the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_stc_wf
            wf = init_bold_stc_wf(
                metadata={"RepetitionTime": 2.0,
                          "SliceTiming": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                )

    Parameters
    ----------
    metadata : :obj:`dict`
        BIDS metadata for BOLD file
    name : :obj:`str`
        Name of workflow (default: ``bold_stc_wf``)

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    skip_vols
        Number of non-steady-state volumes detected at beginning of ``bold_file``

    Outputs
    -------
    stc_file
        Slice-timing corrected BOLD series NIfTI file

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.header import CopyXForm

    slice_times = metadata["SliceTiming"]
    first, last = min(slice_times), max(slice_times)
    frac = config.workflow.slice_time_ref
    tzero = np.round(first + frac * (last - first), 3)

    afni_ver = ''.join('%02d' % v for v in afni.Info().version() or [])
    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
BOLD runs were slice-time corrected to {tzero:0.3g}s ({frac:g} of slice acquisition range
{first:.3g}s-{last:.3g}s) using `3dTshift` from AFNI {afni_ver} [@afni, RRID:SCR_005927].
"""
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file', 'skip_vols']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['stc_file']), name='outputnode')

    LOGGER.log(25, f'BOLD series will be slice-timing corrected to an offset of {tzero:.3g}s.')

    # It would be good to fingerprint memory use of afni.TShift
    slice_timing_correction = pe.Node(
        TShift(outputtype='NIFTI_GZ',
               tr=f"{metadata['RepetitionTime']}s",
               slice_timing=metadata['SliceTiming'],
               slice_encoding_direction=metadata.get('SliceEncodingDirection', 'k'),
               tzero=tzero),
        name='slice_timing_correction')

    copy_xform = pe.Node(CopyXForm(), name='copy_xform', mem_gb=0.1)

    workflow.connect([
        (inputnode, slice_timing_correction, [('bold_file', 'in_file'),
                                              ('skip_vols', 'ignore')]),
        (slice_timing_correction, copy_xform, [('out_file', 'in_file')]),
        (inputnode, copy_xform, [('bold_file', 'hdr_file')]),
        (copy_xform, outputnode, [('out_file', 'stc_file')]),
    ])

    return workflow
