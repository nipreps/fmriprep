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
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.header import ValidateImage
from niworkflows.utils.misc import pass_dummy_scans

DEFAULT_MEMORY_MIN_GB = 0.01


def init_raw_boldref_wf(
    bold_file=None,
    multiecho=False,
    name='raw_boldref_wf',
):
    """
    Build a workflow that generates reference BOLD images for a series.

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    This workflow assumes only one BOLD file has been passed.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.reference import init_raw_boldref_wf
            wf = init_raw_boldref_wf()

    Parameters
    ----------
    bold_file : :obj:`str`
        BOLD series NIfTI file
    multiecho : :obj:`bool`
        If multiecho data was supplied, data from the first echo will be selected
    name : :obj:`str`
        Name of workflow (default: ``bold_reference_wf``)

    Inputs
    ------
    bold_file : str
        BOLD series NIfTI file
    dummy_scans : int or None
        Number of non-steady-state volumes specified by user at beginning of ``bold_file``

    Outputs
    -------
    bold_file : str
        Validated BOLD series NIfTI file
    boldref : str
        Reference image to which BOLD series is motion corrected
    skip_vols : int
        Number of non-steady-state volumes selected at beginning of ``bold_file``
    algo_dummy_scans : int
        Number of non-steady-state volumes agorithmically detected at
        beginning of ``bold_file``

    """
    from niworkflows.interfaces.bold import NonsteadyStatesDetector
    from niworkflows.interfaces.images import RobustAverage

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
First, a reference volume was generated{' from the shortest echo of the BOLD run' * multiecho},
using a custom methodology of *fMRIPrep*, for use in head motion correction.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'dummy_scans']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                'boldref',
                'skip_vols',
                'algo_dummy_scans',
                'validation_report',
            ]
        ),
        name='outputnode',
    )

    # Simplify manually setting input image
    if bold_file is not None:
        inputnode.inputs.bold_file = bold_file

    val_bold = pe.Node(
        ValidateImage(),
        name='val_bold',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    get_dummy = pe.Node(NonsteadyStatesDetector(), name='get_dummy')
    gen_avg = pe.Node(RobustAverage(), name='gen_avg', mem_gb=1)

    calc_dummy_scans = pe.Node(
        niu.Function(function=pass_dummy_scans, output_names=['skip_vols_num']),
        name='calc_dummy_scans',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # fmt: off
    workflow.connect([
        (inputnode, val_bold, [('bold_file', 'in_file')]),
        (inputnode, get_dummy, [('bold_file', 'in_file')]),
        (inputnode, calc_dummy_scans, [('dummy_scans', 'dummy_scans')]),
        (val_bold, gen_avg, [('out_file', 'in_file')]),
        (get_dummy, gen_avg, [('t_mask', 't_mask')]),
        (get_dummy, calc_dummy_scans, [('n_dummy', 'algo_dummy_scans')]),
        (val_bold, outputnode, [
            ('out_file', 'bold_file'),
            ('out_report', 'validation_report'),
        ]),
        (calc_dummy_scans, outputnode, [('skip_vols_num', 'skip_vols')]),
        (gen_avg, outputnode, [('out_file', 'boldref')]),
        (get_dummy, outputnode, [('n_dummy', 'algo_dummy_scans')]),
    ])
    # fmt: on

    return workflow
