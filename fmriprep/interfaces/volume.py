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
    surface = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="the input surface",
    )
    ref_space = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="a volume in the desired output space (dims, spacing, origin)",
    )
    out_vol = File(
        name_source=["surface"],
        name_template="%s.distvol.nii.gz",
        argstr="%s ",
        position=2,
        desc="output - the output volume",
    )
    roi_out = File(
        name_source=["surface"],
        name_template="%s.distvolroi.nii.gz",
        argstr="-roi-out %s ",
        position=3,
        desc="output roi volume of where the output has a computed value",
    )    
    fill_value = traits.Float(
        mandatory=False,
        argstr="-fill-value %f ",
        position=4,
        desc="specify a value to put in all voxels that don't get assigned a distance, default 0",
    )
    exact_limit = traits.Float(
        mandatory=False,
        argstr="-exact-limit %f ",
        position=5,
        desc="specify distance for exact output in mm, default 5",
    )
    approx_limit = traits.Float(
        mandatory=False,
        argstr="-approx-limit %f ",
        position=6,
        desc="specify distance for approximate output in mm, default 20",
    )
    approx_neighborhood = traits.Int(
        mandatory=False,
        argstr="-approx-neighborhood %d ",
        position=7,
        desc="size of neighborhood cube measured from center to face, default 2 = 5x5x5",
    )
    winding_method = traits.Enum(
        "EVEN_ODD",
        "NEGATIVE",
        "NONZERO",
        "NORMALS",    
        argstr="-winding %s ",
        usedefault=True,
        position=8,        
        desc="winding method for point inside surface test, choices: "
             "EVEN_ODD (default) "
             "NEGATIVE "
             "NONZERO "
             "NORMALS "
    )


class CreateSignedDistanceVolumeOutputSpec(TraitedSpec):
    out_vol = File(exists=True, desc="output - the output volume")
    roi_out = File(desc="output roi volume of where the output has a computed value")


class CreateSignedDistanceVolume(WBCommand):
    """
CREATE SIGNED DISTANCE VOLUME FROM SURFACE
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

    >>> from nipype.interfaces.workbench import CreateSignedDistanceVolume
    >>> distvol = CreateSignedDistanceVolume()
    >>> distvol.inputs.surface = 'sub-01.L.pial.native.surf.gii'
    >>> distvol.inputs.refspace = 'sub-01_T1w.nii.gz'
    >>> distvol.inputs.out_vol = 'sub-01.L.pial.native.surf.distvol.nii.gz'
    >>> distvol.inputs.roi_out = 'sub-01.L.pial.native.surf.distvolroi.nii.gz'
    >>> distvol.inputs.fill_value = 0
    >>> distvol.inputs.exact_limit = 5
    >>> distvol.inputs.approx_limit = 20
    >>> distvol.inputs.approx_neighborhood = 2
    >>> distvol.inputs.winding_method = 'EVEN_ODD'
    >>> distvol.cmdline
    'wb_command -create-signed-distance-volume sub-01.L.pial.native.surf.gii \
    sub-01_T1w.nii.gz \
    sub-01.L.pial.native.surf.distvol.nii.gz \
    -roi-out sub-01.L.pial.native.surf.distvolroi.nii.gz \
    -fill-value 0 \
    -exact-limit 5 \
    -approx-limit 20 \
    -approx-neighborhood 2 \
    -winding EVEN_ODD'
    """

    input_spec = CreateSignedDistanceVolumeInputSpec
    output_spec = CreateSignedDistanceVolumeOutputSpec
    _cmd = "wb_command -create-signed-distance-volume"

    def _list_outputs(self):
        outputs = super(CreateSignedDistanceVolume, self)._list_outputs()
        return outputs
