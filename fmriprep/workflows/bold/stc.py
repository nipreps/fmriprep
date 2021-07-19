# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Slice-Timing Correction (STC) of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_stc_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, afni

from ... import config


LOGGER = config.loggers.workflow


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

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
BOLD runs were slice-time corrected using `3dTshift` from
AFNI {afni_ver} [@afni, RRID:SCR_005927].
""".format(afni_ver=''.join(['%02d' % v for v in afni.Info().version() or []]))
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file', 'skip_vols']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['stc_file']), name='outputnode')

    LOGGER.log(25, 'Slice-timing correction will be included.')

    # It would be good to fingerprint memory use of afni.TShift
    slice_timing_correction = pe.Node(
        afni.TShift(outputtype='NIFTI_GZ',
                    tr='{}s'.format(metadata["RepetitionTime"]),
                    slice_timing=metadata['SliceTiming'],
                    slice_encoding_direction=metadata.get('SliceEncodingDirection', 'k')),
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
