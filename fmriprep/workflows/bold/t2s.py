# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
