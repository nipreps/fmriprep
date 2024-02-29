# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""This module provides interfaces for workbench surface commands."""

import os

from nipype import logging
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.workbench.base import WBCommand

iflogger = logging.getLogger('nipype.interface')


class OpenMPTraitedSpec(CommandLineInputSpec):
    num_threads = traits.Int(desc='allows for specifying more threads')


class OpenMPCommandMixin(CommandLine):
    input_spec = OpenMPTraitedSpec

    _num_threads = None

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, 'num_threads')
        if not self._num_threads:
            self._num_threads = os.environ.get('OMP_NUM_THREADS', None)
        if not isdefined(self.inputs.num_threads) and self._num_threads:
            self.inputs.num_threads = int(self._num_threads)
        self._num_threads_update()

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({'OMP_NUM_THREADS': str(self.inputs.num_threads)})

    def run(self, **inputs):
        if 'num_threads' in inputs:
            self.inputs.num_threads = inputs['num_threads']
        self._num_threads_update()
        return super().run(**inputs)


class MetricDilateInputSpec(OpenMPTraitedSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s ',
        position=0,
        desc='the metric to dilate',
    )

    surf_file = File(
        exists=True,
        mandatory=True,
        argstr='%s ',
        position=1,
        desc='the surface to compute on',
    )

    distance = traits.Float(
        mandatory=True,
        argstr='%f ',
        position=2,
        desc='distance in mm to dilate',
    )

    out_file = File(
        name_source=['in_file'],
        name_template='%s.func.gii',
        keep_extension=False,
        argstr='%s ',
        position=3,
        desc='output - the output metric',
    )

    bad_vertex_roi_file = File(
        argstr='-bad-vertex-roi %s ',
        position=4,
        desc='metric file, positive values denote vertices to have their values replaced',
    )

    data_roi_file = File(
        argstr='-data-roi %s ',
        position=5,
        desc='metric file, positive values denote vertices that have data',
    )

    column = traits.Int(
        position=6,
        argstr='-column %d ',
        desc='the column number',
    )

    nearest = traits.Bool(
        position=7,
        argstr='-nearest ',
        desc='use the nearest good value instead of a weighted average',
    )

    linear = traits.Bool(
        position=8,
        argstr='-linear ',
        desc='fill in values with linear interpolation along strongest gradient',
    )

    exponent = traits.Float(
        argstr='-exponent %f ',
        position=9,
        default=6.0,
        desc='exponent n to use in (area / (distance ^ n)) as the '
        'weighting function (default 6)',
    )

    corrected_areas = File(
        argstr='-corrected-areas %s ',
        position=10,
        desc='vertex areas to use instead of computing them from the surface',
    )

    legacy_cutoff = traits.Bool(
        position=11,
        argstr='-legacy-cutoff ',
        desc='use the v1.3.2 method of choosing how many vertices to '
        'use when calculating the dilated value with weighted method',
    )


class MetricDilateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output file')


class MetricDilate(WBCommand, OpenMPCommandMixin):
    """Dilate a metric file on a surface.

    For all data values designated as bad, if they neighbor a good value or
    are within the specified distance of a good value in the same kind of
    model, replace the value with a distance weighted average of nearby good
    values, otherwise set the value to zero.  If -nearest is specified, it
    will use the value from the closest good value within range instead of a
    weighted average.  When the input file contains label data, nearest
    dilation is used on the surface, and weighted popularity is used in the
    volume.

    The -corrected-areas options are intended for dilating on group average
    surfaces, but it is only an approximate correction for the reduction of
    structure in a group average surface.

    If -bad-vertex-roi is specified, all values, including those with
    value zero, are good, except for locations with a positive value in the
    ROI.  If it is not specified, only values equal to zero are bad.
    """

    input_spec = MetricDilateInputSpec
    output_spec = MetricDilateOutputSpec
    _cmd = 'wb_command -metric-dilate '


class MetricResampleInputSpec(OpenMPTraitedSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='The metric file to resample',
    )
    current_sphere = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='A sphere surface with the mesh that the metric is currently on',
    )
    new_sphere = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=2,
        desc='A sphere surface that is in register with <current-sphere> and'
        ' has the desired output mesh',
    )
    method = traits.Enum(
        'ADAP_BARY_AREA',
        'BARYCENTRIC',
        argstr='%s',
        mandatory=True,
        position=3,
        desc='The method name - ADAP_BARY_AREA method is recommended for'
        ' ordinary metric data, because it should use all data while'
        ' downsampling, unlike BARYCENTRIC. If ADAP_BARY_AREA is used,'
        ' exactly one of area_surfs or area_metrics must be specified',
    )
    out_file = File(
        name_source=['new_sphere'],
        name_template='%s.out',
        keep_extension=True,
        argstr='%s',
        position=4,
        desc='The output metric',
    )
    area_surfs = traits.Bool(
        position=5,
        argstr='-area-surfs',
        xor=['area_metrics'],
        desc='Specify surfaces to do vertex area correction based on',
    )
    area_metrics = traits.Bool(
        position=5,
        argstr='-area-metrics',
        xor=['area_surfs'],
        desc='Specify vertex area metrics to do area correction based on',
    )
    current_area = File(
        exists=True,
        position=6,
        argstr='%s',
        desc='A relevant anatomical surface with <current-sphere> mesh OR'
        ' a metric file with vertex areas for <current-sphere> mesh',
    )
    new_area = File(
        exists=True,
        position=7,
        argstr='%s',
        desc='A relevant anatomical surface with <current-sphere> mesh OR'
        ' a metric file with vertex areas for <current-sphere> mesh',
    )
    roi_metric = File(
        exists=True,
        position=8,
        argstr='-current-roi %s',
        desc='Input roi on the current mesh used to exclude non-data vertices',
    )
    valid_roi_out = traits.Bool(
        position=9,
        argstr='-valid-roi-out',
        desc='Output the ROI of vertices that got data from valid source vertices',
    )
    largest = traits.Bool(
        position=10,
        argstr='-largest',
        desc='Use only the value of the vertex with the largest weight',
    )


class MetricResampleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output metric')
    roi_file = File(desc='ROI of vertices that got data from valid source vertices')


class MetricResample(WBCommand, OpenMPCommandMixin):
    """Resample a metric file to a different mesh.

    Resamples a metric file, given two spherical surfaces that are in
    register.  If ``ADAP_BARY_AREA`` is used, exactly one of -area-surfs or
    ``-area-metrics`` must be specified.

    The ``ADAP_BARY_AREA`` method is recommended for ordinary metric data,
    because it should use all data while downsampling, unlike ``BARYCENTRIC``.
    The recommended areas option for most data is individual midthicknesses
    for individual data, and averaged vertex area metrics from individual
    midthicknesses for group average data.

    The ``-current-roi`` option only masks the input, the output may be slightly
    dilated in comparison, consider using ``-metric-mask`` on the output when
    using ``-current-roi``.

    The ``-largest option`` results in nearest vertex behavior when used with
    ``BARYCENTRIC``.  When resampling a binary metric, consider thresholding at
    0.5 after resampling rather than using ``-largest``.
    """

    input_spec = MetricResampleInputSpec
    output_spec = MetricResampleOutputSpec
    _cmd = 'wb_command -metric-resample'

    def _format_arg(self, opt, spec, val):
        if opt in ('current_area', 'new_area'):
            if not self.inputs.area_surfs and not self.inputs.area_metrics:
                raise ValueError(f'{opt} was set but neither area_surfs or area_metrics were set')
        if opt == 'method':
            if (
                val == 'ADAP_BARY_AREA'
                and not self.inputs.area_surfs
                and not self.inputs.area_metrics
            ):
                raise ValueError('Exactly one of area_surfs or area_metrics must be specified')
        if opt == 'valid_roi_out' and val:
            # generate a filename and add it to argstr
            roi_out = self._gen_filename(self.inputs.in_file, suffix='_roi')
            iflogger.info('Setting roi output file as', roi_out)
            spec.argstr += ' ' + roi_out
        return super()._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = super()._list_outputs()
        if self.inputs.valid_roi_out:
            roi_file = self._gen_filename(self.inputs.in_file, suffix='_roi')
            outputs['roi_file'] = os.path.abspath(roi_file)
        return outputs


class VolumeToSurfaceMappingInputSpec(OpenMPTraitedSpec):
    volume_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=1,
        desc='the volume to map data from',
    )
    surface_file = File(
        exists=True,
        argstr='%s',
        mandatory=True,
        position=2,
        desc='the surface to map the data onto',
    )
    out_file = File(
        name_source=['surface_file'],
        name_template='%s_mapped.func.gii',
        keep_extension=False,
        argstr='%s',
        position=3,
        desc='the output metric file',
    )
    method = traits.Enum(
        'trilinear',
        'enclosing',
        'cubic',
        'ribbon-constrained',
        'myelin-style',
        argstr='-%s',
        position=4,
        desc='the interpolation method to use',
    )
    _ribbon_constrained = [
        'inner_surface',
        'outer_surface',
        'volume_roi',
        'weighted',
        'voxel_subdiv',
        'gaussian',
        'interpolate',
        'bad_vertices_out',
        'output_weights',
        'output_weights_text',
    ]
    _myelin_style = [
        'ribbon_roi',
        'thickness',
        'sigma',
        'legacy_bug',
    ]
    inner_surface = File(
        exists=True,
        argstr='%s',
        position=5,
        desc='the inner surface of the ribbon [-ribbon-constrained]',
        xor=_myelin_style,
    )
    outer_surface = File(
        exists=True,
        argstr='%s',
        position=6,
        desc='the outer surface of the ribbon [-ribbon-constrained]',
        xor=_myelin_style,
    )
    volume_roi = File(
        exists=True,
        argstr='-volume-roi %s',
        position=7,
        desc='use a volume roi [-ribbon-constrained]',
        xor=_myelin_style,
    )
    weighted = traits.Bool(
        argstr='-weighted',
        position=8,
        desc='treat the roi values as weightings rather than binary [-ribbon-constrained]',
        requires=['volume_roi'],
        xor=_myelin_style,
    )
    voxel_subdiv = traits.Int(
        default_value=3,
        argstr='-voxel-subdiv %d',
        desc='voxel divisions while estimating voxel weights [-ribbon-constrained]',
        xor=_myelin_style,
    )
    thin_columns = traits.Bool(
        argstr='-thin-columns',
        desc='use non-overlapping polyhedra [-ribbon-constrained]',
        xor=_myelin_style,
    )
    gaussian = traits.Float(
        argstr='-gaussian %g',
        desc="reduce weight to voxels that aren't near <surface> [-ribbon-constrained]",
        xor=_myelin_style,
    )
    interpolate = traits.Enum(
        'CUBIC',
        'TRILINEAR',
        'ENCLOSING_VOXEL',
        argstr='-interpolate %s',
        desc='instead of a weighted average of voxels, '
        'interpolate at subpoints inside the ribbon [-ribbon-constrained]',
        xor=_myelin_style,
    )
    bad_vertices_out = File(
        argstr='-bad-vertices-out %s',
        desc="output an ROI of which vertices didn't intersect any valid voxels",
        xor=_myelin_style,
    )
    output_weights = traits.Int(
        argstr='-output-weights %(0)d output_weights.nii.gz',
        desc='write the voxel weights for a vertex to a volume file',
        xor=_myelin_style,
    )
    output_weights_text = traits.File(
        argstr='-output-weights-text %s',
        desc='write the voxel weights for all vertices to a text file',
        xor=_myelin_style,
    )
    ribbon_roi = File(
        exists=True,
        argstr='%s',
        position=5,
        desc='an roi volume of the cortical ribbon for this hemisphere [-myelin-style]',
        xor=_ribbon_constrained,
    )
    thickness = File(
        exists=True,
        argstr='%s',
        position=6,
        desc='the thickness metric file for this hemisphere [-myelin-style]',
        xor=_ribbon_constrained,
    )
    sigma = traits.Float(
        argstr='%g',
        position=7,
        desc='gaussian kernel in mm for weighting voxels within range [-myelin-style]',
        xor=_ribbon_constrained,
    )
    legacy_bug = traits.Bool(
        argstr='-legacy-bug',
        position=8,
        desc='use the old bug in the myelin-style algorithm [-myelin-style]',
        xor=_ribbon_constrained,
    )
    subvol_select = traits.Int(
        argstr='-subvol-select %d',
        desc='select a single subvolume to map',
    )

    """\
MAP VOLUME TO SURFACE
   wb_command -volume-to-surface-mapping
      <volume> - the volume to map data from
      <surface> - the surface to map the data onto
      <metric-out> - output - the output metric file

      [-trilinear] - use trilinear volume interpolation

      [-enclosing] - use value of the enclosing voxel

      [-cubic] - use cubic splines

      [-ribbon-constrained] - use ribbon constrained mapping algorithm
         <inner-surf> - the inner surface of the ribbon
         <outer-surf> - the outer surface of the ribbon

         [-volume-roi] - use a volume roi
            <roi-volume> - the roi volume file

            [-weighted] - treat the roi values as weightings rather than binary

         [-voxel-subdiv] - voxel divisions while estimating voxel weights
            <subdiv-num> - number of subdivisions, default 3

         [-thin-columns] - use non-overlapping polyhedra

         [-gaussian] - reduce weight to voxels that aren't near <surface>
            <scale> - value to multiply the local thickness by, to get the
               gaussian sigma

         [-interpolate] - instead of a weighted average of voxels, interpolate
            at subpoints inside the ribbon
            <method> - interpolation method, must be CUBIC, ENCLOSING_VOXEL, or
               TRILINEAR

         [-bad-vertices-out] - output an ROI of which vertices didn't intersect
            any valid voxels
            <roi-out> - output - the output metric file of vertices that have
               no data

         [-output-weights] - write the voxel weights for a vertex to a volume
            file
            <vertex> - the vertex number to get the voxel weights for, 0-based
            <weights-out> - output - volume to write the weights to

         [-output-weights-text] - write the voxel weights for all vertices to a
            text file
            <text-out> - output - the output text filename

      [-myelin-style] - use the method from myelin mapping
         <ribbon-roi> - an roi volume of the cortical ribbon for this
            hemisphere
         <thickness> - a metric file of cortical thickness
         <sigma> - gaussian kernel in mm for weighting voxels within range

         [-legacy-bug] - emulate old v1.2.3 and earlier code that didn't follow
            a cylinder cutoff

      [-subvol-select] - select a single subvolume to map
         <subvol> - the subvolume number or name
"""


class VolumeToSurfaceMappingOutputSpec(TraitedSpec):
    out_file = File(desc='the output metric file')
    bad_vertices_file = File(desc='the output metric file of vertices that have no data')
    weights_file = File(desc='volume to write the weights to')
    weights_text_file = File(desc='the output text filename')


class VolumeToSurfaceMapping(WBCommand, OpenMPCommandMixin):
    """Map a volume to a surface using one of several methods.

    From https://humanconnectome.org/software/workbench-command/-volume-to-surface-mapping::

        You must specify exactly one mapping method.  Enclosing voxel uses the
        value from the voxel the vertex lies inside, while trilinear does a 3D
        linear interpolation based on the voxels immediately on each side of the
        vertex's position.

        The ribbon mapping method constructs a polyhedron from the vertex's
        neighbors on each surface, and estimates the amount of this polyhedron's
        volume that falls inside any nearby voxels, to use as the weights for
        sampling.  If -thin-columns is specified, the polyhedron uses the edge
        midpoints and triangle centroids, so that neighboring vertices do not
        have overlapping polyhedra.  This may require increasing -voxel-subdiv to
        get enough samples in each voxel to reliably land inside these smaller
        polyhedra.  The volume ROI is useful to exclude partial volume effects of
        voxels the surfaces pass through, and will cause the mapping to ignore
        voxels that don't have a positive value in the mask.  The subdivision
        number specifies how it approximates the amount of the volume the
        polyhedron intersects, by splitting each voxel into NxNxN pieces, and
        checking whether the center of each piece is inside the polyhedron.  If
        you have very large voxels, consider increasing this if you get zeros in
        your output.  The -gaussian option makes it act more like the myelin
        method, where the distance of a voxel from <surface> is used to
        downweight the voxel.  The -interpolate suboption, instead of doing a
        weighted average of voxels, interpolates from the volume at the
        subdivided points inside the ribbon.  If using both -interpolate and the
        -weighted suboption to -volume-roi, the roi volume weights are linearly
        interpolated, unless the -interpolate method is ENCLOSING_VOXEL, in which
        case ENCLOSING_VOXEL is also used for sampling the roi volume weights.

        The myelin style method uses part of the caret5 myelin mapping command to
        do the mapping: for each surface vertex, take all voxels that are in a
        cylinder with radius and height equal to cortical thickness, centered on
        the vertex and aligned with the surface normal, and that are also within
        the ribbon ROI, and apply a gaussian kernel with the specified sigma to
        them to get the weights to use.  The -legacy-bug flag reverts to the
        unintended behavior present from the initial implementation up to and
        including v1.2.3, which had only the tangential cutoff and a bounding box
        intended to be larger than where the cylinder cutoff should have been.

    Examples:
    >>> from fmriprep.interfaces.workbench import VolumeToSurfaceMapping
    >>> vol2surf = VolumeToSurfaceMapping()
    >>> vol2surf.inputs.volume_file = 'bold.nii.gz'
    >>> vol2surf.inputs.surface_file = 'lh.midthickness.surf.gii'
    >>> vol2surf.inputs.method = 'ribbon-constrained'
    >>> vol2surf.inputs.inner_surface = 'lh.white.surf.gii'
    >>> vol2surf.inputs.outer_surface = 'lh.pial.surf.gii'
    >>> vol2surf.cmdline  # doctest: +NORMALIZE_WHITESPACE
    'wb_command -volume-to-surface-mapping bold.nii.gz lh.midthickness.surf.gii \
    lh.midthickness.surf_mapped.func.gii -ribbon-constrained lh.white.surf.gii lh.pial.surf.gii'
    """

    input_spec = VolumeToSurfaceMappingInputSpec
    output_spec = VolumeToSurfaceMappingOutputSpec
    _cmd = 'wb_command -volume-to-surface-mapping'

    def _format_arg(self, opt, spec, val):
        if opt in self.input_spec._ribbon_constrained:
            if self.inputs.method != 'ribbon-constrained':
                return ''
        elif opt in self.input_spec._myelin_style:
            if self.inputs.method != 'myelin-style':
                return ''
        return super()._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = super()._list_outputs()
        if isdefined(self.inputs.bad_vertices_out):
            outputs['bad_vertices_file'] = os.path.abspath(self.inputs.bad_vertices_out)
        if isdefined(self.inputs.output_weights):
            outputs['weights_file'] = os.path.abspath(self.inputs.output_weights)
        if isdefined(self.inputs.output_weights_text):
            outputs['weights_text_file'] = os.path.abspath(self.inputs.output_weights_text)
        return outputs


class MetricMaskInputSpec(CommandLineInputSpec):
    """MASK A METRIC FILE
    wb_command -metric-mask
       <metric> - the input metric
       <mask> - the mask metric
       <metric-out> - output - the output metric

       [-column] - select a single column
          <column> - the column number or name

       By default, the output metric is a copy of the input metric, but with
       zeros wherever the mask metric is zero or negative.  if -column is
       specified, the output contains only one column, the masked version of the
       specified input column."""

    in_file = File(
        exists=True,
        argstr='%s',
        position=1,
        mandatory=True,
        desc='input metric file',
    )
    mask = File(
        exists=True,
        argstr='%s',
        position=2,
        mandatory=True,
        desc='mask metric file',
    )
    out_file = File(
        name_template='%s_masked.func.gii',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=3,
        desc='output metric file',
    )
    column = traits.Either(
        traits.Int,
        traits.String,
        argstr='-column %s',
        desc='select a single column by number or name',
    )


class MetricMaskOutputSpec(TraitedSpec):
    out_file = File(desc='output metric file')


class MetricMask(WBCommand):
    """Mask a metric file.

    Examples

    >>> from fmriprep.interfaces.workbench import MetricMask
    >>> metric_mask = MetricMask()
    >>> metric_mask.inputs.in_file = 'lh.bold.func.gii'
    >>> metric_mask.inputs.mask = 'lh.roi.shape.gii'
    >>> metric_mask.cmdline
    'wb_command -metric-mask lh.bold.func.gii lh.roi.shape.gii lh.bold.func_masked.func.gii'
    """

    input_spec = MetricMaskInputSpec
    output_spec = MetricMaskOutputSpec
    _cmd = 'wb_command -metric-mask'


class MetricFillHolesInputSpec(TraitedSpec):
    """FILL HOLES IN AN ROI METRIC

    wb_command -metric-fill-holes
       <surface> - the surface to use for neighbor information
       <metric-in> - the input ROI metric
       <metric-out> - output - the output ROI metric

       [-corrected-areas] - vertex areas to use instead of computing them from
          the surface
          <area-metric> - the corrected vertex areas, as a metric

       Finds all connected areas that are not included in the ROI, and writes
       ones into all but the largest one, in terms of surface area."""

    surface_file = File(
        mandatory=True,
        exists=True,
        argstr='%s',
        position=1,
        desc='surface to use for neighbor information',
    )
    metric_file = File(
        mandatory=True,
        exists=True,
        argstr='%s',
        position=2,
        desc='input ROI metric',
    )
    out_file = File(
        name_template='%s_filled.shape.gii',
        name_source='metric_file',
        keep_extension=False,
        argstr='%s',
        position=3,
        desc='output ROI metric',
    )
    corrected_areas = File(
        exists=True,
        argstr='-corrected-areas %s',
        desc='vertex areas to use instead of computing them from the surface',
    )


class MetricFillHolesOutputSpec(TraitedSpec):
    out_file = File(desc='output ROI metric')


class MetricFillHoles(WBCommand):
    """Fill holes in an ROI metric.

    Examples

    >>> from fmriprep.interfaces.workbench import MetricFillHoles
    >>> fill_holes = MetricFillHoles()
    >>> fill_holes.inputs.surface_file = 'lh.midthickness.surf.gii'
    >>> fill_holes.inputs.metric_file = 'lh.roi.shape.gii'
    >>> fill_holes.cmdline  # doctest: +NORMALIZE_WHITESPACE
    'wb_command -metric-fill-holes lh.midthickness.surf.gii lh.roi.shape.gii \
    lh.roi.shape_filled.shape.gii'
    """

    input_spec = MetricFillHolesInputSpec
    output_spec = MetricFillHolesOutputSpec
    _cmd = 'wb_command -metric-fill-holes'


class MetricRemoveIslandsInputSpec(TraitedSpec):
    """REMOVE ISLANDS IN AN ROI METRIC

    wb_command -metric-remove-islands
       <surface> - the surface to use for neighbor information
       <metric-in> - the input ROI metric
       <metric-out> - output - the output ROI metric

       [-corrected-areas] - vertex areas to use instead of computing them from
          the surface
          <area-metric> - the corrected vertex areas, as a metric

    Finds all connected areas in the ROI, and zeros out all but the largest
    one, in terms of surface area."""

    surface_file = File(
        mandatory=True,
        exists=True,
        argstr='%s',
        position=1,
        desc='surface to use for neighbor information',
    )
    metric_file = File(
        mandatory=True,
        exists=True,
        argstr='%s',
        position=2,
        desc='input ROI metric',
    )
    out_file = File(
        name_template='%s_noislands.shape.gii',
        name_source='metric_file',
        keep_extension=False,
        argstr='%s',
        position=3,
        desc='output ROI metric',
    )
    corrected_areas = File(
        exists=True,
        argstr='-corrected-areas %s',
        desc='vertex areas to use instead of computing them from the surface',
    )


class MetricRemoveIslandsOutputSpec(TraitedSpec):
    out_file = File(desc='output ROI metric')


class MetricRemoveIslands(WBCommand):
    """Remove islands in an ROI metric.

    Examples

    >>> from fmriprep.interfaces.workbench import MetricRemoveIslands
    >>> remove_islands = MetricRemoveIslands()
    >>> remove_islands.inputs.surface_file = 'lh.midthickness.surf.gii'
    >>> remove_islands.inputs.metric_file = 'lh.roi.shape.gii'
    >>> remove_islands.cmdline  # doctest: +NORMALIZE_WHITESPACE
    'wb_command -metric-remove-islands lh.midthickness.surf.gii \
    lh.roi.shape.gii lh.roi.shape_noislands.shape.gii'
    """

    input_spec = MetricRemoveIslandsInputSpec
    output_spec = MetricRemoveIslandsOutputSpec
    _cmd = 'wb_command -metric-remove-islands'
