# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""This module provides interfaces for workbench sphere projection commands"""

# from distutils.cmd import Command
from signal import valid_signals
from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import (
    TraitedSpec,
    File,
    traits,
    CommandLineInputSpec,
    SimpleInterface,
)
from nipype import logging


class SurfaceSphereProjectUnprojectInputSpec(CommandLineInputSpec):

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="a sphere with the desired output mesh",
    )

    sphere_project_to = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="a sphere that aligns with sphere-in",
    )

    sphere_unproject_from = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=2,
        desc="deformed to the desired output space",
    )

    out_file = File(
        name_source="in_file",
        name_template="%s_deformed.surf.gii ",
        keep_extension=False,
        argstr="%s",
        position=3,
        desc="The sphere output file",
    )


class SurfaceSphereProjectUnprojectOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output file")


class SurfaceSphereProjectUnproject(WBCommand):
    # COPY REGISTRATION DEFORMATIONS TO DIFFERENT SPHERE
    # wb_command -surface-sphere-project-unproject
    # <sphere-in> - a sphere with the desired output mesh
    # <sphere-project-to> - a sphere that aligns with sphere-in
    # <sphere-unproject-from> - <sphere-project-to> deformed to the desired output space
    # <sphere-out> - output - the output sphere

    input_spec = SurfaceSphereProjectUnprojectInputSpec
    output_spec = SurfaceSphereProjectUnprojectOutputSpec
    _cmd = "wb_command -surface-sphere-project-unproject "