# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""This module provides interfaces for workbench volume commands"""
import os

from nipype.interfaces.base import TraitedSpec, File, traits, CommandLineInputSpec
from nipype.interfaces.workbench.base import WBCommand
from nipype import logging

iflogger = logging.getLogger("nipype.interface")


class CreateSignedDistanceVolumeInputSpec(CommandLineInputSpec):
    surf_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="Input surface GIFTI file (.surf.gii)",
    )
    ref_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="NIfTI volume in the desired output space (dims, spacing, origin)",
    )
    out_file = File(
        name_source=["surface"],
        name_template="%s_distvol.nii.gz",
        argstr="%s",
        position=2,
        desc="Name of output volume containing signed distances",
    )
    out_mask = File(
        name_source=["surface"],
        name_template="%s_distmask.nii.gz",
        argstr="-roi-out %s",
        desc="Name of file to store a mask where the ``out_file`` has a computed value",
    )    
    fill_value = traits.Float(
        mandatory=False,
        default=0.,
        usedefault=True,
        argstr="-fill-value %f",
        desc="value to put in all voxels that don't get assigned a distance",
    )
    exact_limit = traits.Float(
        default=5.,
        usedefault=True,
        argstr="-exact-limit %f",
        desc="distance for exact output in mm",
    )
    approx_limit = traits.Float(
        default=20.,
        usedefault=True,
        argstr="-approx-limit %f",
        desc="distance for approximate output in mm",
    )
    approx_neighborhood = traits.Int(
        default=2,
        usedefault=True,
        argstr="-approx-neighborhood %d",
        desc="size of neighborhood cube measured from center to face in voxels, default 2 = 5x5x5",
    )
    winding_method = traits.Enum(
        "EVEN_ODD",
        "NEGATIVE",
        "NONZERO",
        "NORMALS",    
        argstr="-winding %s",
        usedefault=True,
        desc="winding method for point inside surface test"
    )

class CreateSignedDistanceVolumeOutputSpec(TraitedSpec):
    out_file = File(desc="Name of output volume containing signed distances")
    out_mask = File(desc="Name of file to store a mask where the ``out_file`` has a computed value")


class CreateSignedDistanceVolume(WBCommand):
    """CREATE SIGNED DISTANCE VOLUME FROM SURFACE
   wb_command -create-signed-distance-volume
      <surface> - the input surface
      <refspace> - a volume in the desired output space (dims, spacing, origin)
      <outvol> - output - the output volume

      [-roi-out] - output an roi volume of where the output has a computed
         value
         <roi-vol> - output - the output roi volume

      [-fill-value] - specify a value to put in all voxels that don't get
         assigned a distance
         <value> - value to fill with (default 0)

      [-exact-limit] - specify distance for exact output
         <dist> - distance in mm (default 5)

      [-approx-limit] - specify distance for approximate output
         <dist> - distance in mm (default 20)

      [-approx-neighborhood] - voxel neighborhood for approximate calculation
         <num> - size of neighborhood cube measured from center to face, in
            voxels (default 2 = 5x5x5)

      [-winding] - winding method for point inside surface test
         <method> - name of the method (default EVEN_ODD)

      Computes the signed distance function of the surface.  Exact distance is
      calculated by finding the closest point on any surface triangle to the
      center of the voxel.  Approximate distance is calculated starting with
      these distances, using dijkstra's method with a neighborhood of voxels.
      Specifying too small of an exact distance may produce unexpected results.
      Valid specifiers for winding methods are as follows:

      EVEN_ODD (default)
      NEGATIVE
      NONZERO
      NORMALS

      The NORMALS method uses the normals of triangles and edges, or the
      closest triangle hit by a ray from the point.  This method may be
      slightly faster, but is only reliable for a closed surface that does not
      cross through itself.  All other methods count entry (positive) and exit
      (negative) crossings of a vertical ray from the point, then counts as
      inside if the total is odd, negative, or nonzero, respectively.

    """

    input_spec = CreateSignedDistanceVolumeInputSpec
    output_spec = CreateSignedDistanceVolumeOutputSpec
    _cmd = "wb_command -create-signed-distance-volume"