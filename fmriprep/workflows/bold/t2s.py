# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generate T2* map from multi-echo BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_t2s_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.func.util import init_skullstrip_bold_wf

from ...interfaces import T2SMap
from ... import config


LOGGER = config.loggers.workflow


# pylint: disable=R0914
def init_bold_t2s_wf(echo_times, mem_gb, omp_nthreads,
                     t2s_coreg=False, name='bold_t2s_wf'):
    """
    Combine multiple echos of :abbr:`ME-EPI (multi-echo echo-planar imaging)`.

    This workflow wraps the `tedana`_ `T2* workflow`_ to optimally
    combine multiple echos and derive a T2* map for optional use as a
    coregistration target.
    The following steps are performed:

    #. :abbr:`HMC (head motion correction)` on individual echo files.
    #. Compute the T2* map
    #. Create an optimally combined ME-EPI time series

    .. _tedana: https://github.com/me-ica/tedana
    .. _`T2* workflow`: https://tedana.readthedocs.io/en/latest/generated/tedana.workflows.t2smap_workflow.html#tedana.workflows.t2smap_workflow  # noqa

    Parameters
    ----------
    echo_times : :obj:`list`
        list of TEs associated with each echo
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    t2s_coreg : :obj:`bool`
        Use the calculated T2*-map for T2*-driven coregistration
    name : :obj:`str`
        Name of workflow (default: ``bold_t2s_wf``)

    Inputs
    ------
    bold_file
        list of individual echo files

    Outputs
    -------
    bold
        the optimally combined time series for all supplied echos
    bold_mask
        the binarized, skull-stripped adaptive T2* map
    bold_ref_brain
        the adaptive T2* map

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A T2* map was estimated from the preprocessed BOLD by fitting to a monoexponential signal
decay model with log-linear regression.
For each voxel, the maximal number of echoes with reliable signal in that voxel were
used to fit the model.
The calculated T2* map was then used to optimally combine preprocessed BOLD across
echoes following the method described in [@posse_t2s].
The optimally combined time series was carried forward as the *preprocessed BOLD*{}.
""".format('' if not t2s_coreg else ', and the T2* map was also retained as the BOLD reference')

    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['bold', 'bold_mask', 'bold_ref_brain']),
                         name='outputnode')

    LOGGER.log(25, 'Generating T2* map and optimally combined ME-EPI time series.')

    t2smap_node = pe.Node(T2SMap(echo_times=echo_times), name='t2smap_node')
    skullstrip_t2smap_wf = init_skullstrip_bold_wf(name='skullstrip_t2smap_wf')

    workflow.connect([
        (inputnode, t2smap_node, [('bold_file', 'in_files')]),
        (t2smap_node, outputnode, [('optimal_comb', 'bold')]),
        (t2smap_node, skullstrip_t2smap_wf, [('t2star_map', 'inputnode.in_file')]),
        (skullstrip_t2smap_wf, outputnode, [
            ('outputnode.mask_file', 'bold_mask'),
            ('outputnode.skull_stripped_file', 'bold_ref_brain')]),
    ])

    return workflow
