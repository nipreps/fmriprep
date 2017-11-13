# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Slice-Timing Correction (STC) of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_stc_wf

"""
from niworkflows.nipype import logging
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import utility as niu, afni
from niworkflows.interfaces.utils import CopyXForm

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('workflow')


# pylint: disable=R0914
def init_bold_stc_wf(metadata, name='bold_stc_wf'):
    """
    This workflow performs :abbr:`STC (slice-timing correction)` over the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_stc_wf
        wf = init_bold_stc_wf(
            metadata={"RepetitionTime": 2.0,
                      "SliceTiming": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            )

    **Parameters**

        metadata : dict
            BIDS metadata for BOLD file
        name : str
            Name of workflow (default: ``bold_stc_wf``)

    **Inputs**

        bold_file
            BOLD series NIfTI file
        skip_vols
            Number of non-steady-state volumes detected at beginning of ``bold_file``

    **Outputs**

        stc_file
            Slice-timing corrected BOLD series NIfTI file

    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file', 'skip_vols']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['stc_file']), name='outputnode')

    LOGGER.log(25, 'Slice-timing correction will be included.')

    def create_custom_slice_timing_file_func(metadata):
        import os
        slice_timings = metadata["SliceTiming"]
        slice_timings_ms = [str(t) for t in slice_timings]
        out_file = "timings.1D"
        with open("timings.1D", "w") as fp:
            fp.write("\t".join(slice_timings_ms))

        return os.path.abspath(out_file)

    create_custom_slice_timing_file = pe.Node(
        niu.Function(function=create_custom_slice_timing_file_func),
        name="create_custom_slice_timing_file",
        mem_gb=DEFAULT_MEMORY_MIN_GB)
    create_custom_slice_timing_file.inputs.metadata = metadata

    # It would be good to fingerprint memory use of afni.TShift
    slice_timing_correction = pe.Node(
        afni.TShift(outputtype='NIFTI_GZ', tr='{}s'.format(metadata["RepetitionTime"])),
        name='slice_timing_correction')

    copy_xform = pe.Node(CopyXForm(), name='copy_xform', mem_gb=0.1, run_without_submitting=True)

    def _prefix_at(x):
        return "@" + x

    workflow.connect([
        (inputnode, slice_timing_correction, [('bold_file', 'in_file'),
                                              ('skip_vols', 'ignore')]),
        (create_custom_slice_timing_file, slice_timing_correction, [
            (('out', _prefix_at), 'tpattern')]),
        (slice_timing_correction, copy_xform, [('out_file', 'in_file')]),
        (inputnode, copy_xform, [('bold_file', 'hdr_file')]),
        (copy_xform, outputnode, [('out_file', 'stc_file')]),
    ])

    return workflow
