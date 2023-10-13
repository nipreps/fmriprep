from __future__ import annotations

import os
import typing as ty

import nibabel as nb
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.utility import KeySelect
from niworkflows.utils.connections import listify

from fmriprep import config

from ...interfaces.resampling import (
    DistortionParameters,
    ReconstructFieldmap,
    ResampleSeries,
)
from ...utils.misc import estimate_bold_mem_usage
from .stc import init_bold_stc_wf
from .t2s import init_bold_t2s_wf, init_t2s_reporting_wf

if ty.TYPE_CHECKING:
    from niworkflows.utils.spaces import SpatialReferences


def init_bold_apply_wf(
    *,
    spaces: SpatialReferences,
    name: str = 'bold_apply_wf',
) -> pe.Workflow:
    """TODO: Docstring"""
    from smriprep.workflows.outputs import init_template_iterator_wf

    workflow = pe.Workflow(name=name)

    if getattr(spaces, "_cached") is not None and spaces.cached.references:
        template_iterator_wf = init_template_iterator_wf(spaces=spaces)
        # TODO: Refactor bold_std_trans_wf
        # bold_std_trans_wf = init_bold_std_trans_wf(
        #     freesurfer=config.workflow.run_reconall,
        #     mem_gb=mem_gb["resampled"],
        #     omp_nthreads=config.nipype.omp_nthreads,
        #     spaces=spaces,
        #     multiecho=multiecho,
        #     use_compression=not config.execution.low_mem,
        #     name="bold_std_trans_wf",
        # )

    return workflow
