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
Generate T2* map from multi-echo BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_t2s_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...interfaces.multiecho import T2SMap
from ...interfaces.maths import Label2Mask, Clip
from ...interfaces.reports import LabeledHistogram
from ... import config


LOGGER = config.loggers.workflow


# pylint: disable=R0914
def init_bold_t2s_wf(echo_times, mem_gb, omp_nthreads,
                     name='bold_t2s_wf'):
    r"""
    Combine multiple echos of :abbr:`ME-EPI (multi-echo echo-planar imaging)`.

    This workflow wraps the `tedana`_ `T2* workflow`_ to optimally
    combine multiple preprocessed echos and derive a T2\ :sup:`★` map.
    The following steps are performed:
    #. Compute the T2\ :sup:`★` map
    #. Create an optimally combined ME-EPI time series

    .. _tedana: https://github.com/me-ica/tedana
    .. _`T2* workflow`: https://tedana.readthedocs.io/en/latest/generated/tedana.workflows.t2smap_workflow.html#tedana.workflows.t2smap_workflow  # noqa

    Parameters
    ----------
    echo_times : :obj:`list` or :obj:`tuple`
        list of TEs associated with each echo
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``bold_t2s_wf``)

    Inputs
    ------
    bold_file
        list of individual echo files
    bold_mask
        a binary mask to apply to the BOLD files

    Outputs
    -------
    bold
        the optimally combined time series for all supplied echos
    t2star_map
        the calculated T2\ :sup:`★` map

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A T2<sup>★</sup> map was estimated from the preprocessed EPI echoes, by voxel-wise fitting
the maximal number of echoes with reliable signal in that voxel to a monoexponential signal
decay model with nonlinear regression.
The T2<sup>★</sup>/S<sub>0</sub> estimates from a log-linear regression fit were used for
initial values.
The calculated T2<sup>★</sup> map was then used to optimally combine preprocessed BOLD across
echoes following the method described in [@posse_t2s].
The optimally combined time series was carried forward as the *preprocessed BOLD*.
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file', 'bold_mask']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['bold', 't2star_map']), name='outputnode')

    LOGGER.log(25, 'Generating T2* map and optimally combined ME-EPI time series.')

    t2smap_node = pe.Node(T2SMap(echo_times=list(echo_times)), name='t2smap_node')

    workflow.connect([
        (inputnode, t2smap_node, [('bold_file', 'in_files'),
                                  ('bold_mask', 'mask_file')]),
        (t2smap_node, outputnode, [('optimal_comb', 'bold'),
                                   ('t2star_map', 't2star_map')]),
    ])

    return workflow


def init_t2s_reporting_wf(name='t2s_reporting_wf'):
    r"""
    Generate T2\*-map reports.

    This workflow generates a histogram of esimated T2\* values (in seconds) in the
    cortical and subcortical gray matter mask.

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``t2s_reporting_wf``)

    Inputs
    ------
    t2star_file
        estimated T2\* map
    boldref
        reference BOLD file
    label_file
        an integer label file identifying gray matter with value ``1``
    label_bold_xform
        Affine matrix that maps the label file into alignment with the native
        BOLD space; can be ``"identity"`` if label file is already aligned

    Outputs
    -------
    t2star_hist
        an SVG histogram showing estimated T2\* values in gray matter
    t2s_comp_report
        a before/after figure comparing the reference BOLD image and T2\* map
    """
    from nipype.pipeline import engine as pe
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.interfaces.reportlets.registration import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t2star_file', 'boldref', 'label_file', 'label_bold_xform']
        ),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['t2star_hist', 't2s_comp_report']),
        name='outputnode')

    label_tfm = pe.Node(ApplyTransforms(interpolation="MultiLabel"), name="label_tfm")

    gm_mask = pe.Node(Label2Mask(label_val=1), name="gm_mask")

    clip_t2star = pe.Node(Clip(maximum=0.1), name="clip_t2star")

    t2s_hist = pe.Node(LabeledHistogram(mapping={1: "Gray matter"}, xlabel='T2* (s)'),
                       name='t2s_hist')

    t2s_comparison = pe.Node(
        SimpleBeforeAfter(
            before_label="BOLD Reference",
            after_label="T2* Map",
            dismiss_affine=True,
        ),
        name="t2s_comparison",
        mem_gb=0.1,
    )

    workflow.connect([
        (inputnode, label_tfm, [('label_file', 'input_image'),
                                ('t2star_file', 'reference_image'),
                                ('label_bold_xform', 'transforms')]),
        (inputnode, clip_t2star, [('t2star_file', 'in_file')]),
        (clip_t2star, t2s_hist, [('out_file', 'in_file')]),
        (label_tfm, gm_mask, [('output_image', 'in_file')]),
        (gm_mask, t2s_hist, [('out_file', 'label_file')]),
        (inputnode, t2s_comparison, [('boldref', 'before'),
                                     ('t2star_file', 'after')]),
        (gm_mask, t2s_comparison, [('out_file', 'wm_seg')]),
        (t2s_hist, outputnode, [('out_report', 't2star_hist')]),
        (t2s_comparison, outputnode, [('out_report', 't2s_comp_report')]),
    ])

    return workflow
