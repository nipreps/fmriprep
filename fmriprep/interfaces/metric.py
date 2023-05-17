# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""This module provides interfaces for workbench surface commands"""
import os

from nipype import logging
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, traits
from nipype.interfaces.workbench.base import WBCommand

iflogger = logging.getLogger("nipype.interface")


class MetricDilateInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="the metric to dilate",
    )

    surf_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="the surface to compute on",
    )

    distance = traits.Float(
        mandatory=True,
        argstr="%f ",
        position=2,
        desc="distance in mm to dilate",
    )

    out_file = File(
        name_source=["in_file"],
        name_template="%s.func.gii",
        keep_extension=False,
        argstr="%s ",
        position=3,
        desc="output - the output metric",
    )

    bad_vertex_roi_file = File(
        argstr="-bad-vertex-roi %s ",
        position=4,
        desc="metric file, positive values denote vertices to have their values replaced",
    )

    data_roi_file = File(
        argstr="-data-roi %s ",
        position=5,
        desc="metric file, positive values denote vertices that have data",
    )

    column = traits.Int(
        position=6,
        argstr="-column %d ",
        desc="the column number",
    )

    nearest = traits.Bool(
        position=7,
        argstr="-nearest ",
        desc="use the nearest good value instead of a weighted average",
    )

    linear = traits.Bool(
        position=8,
        argstr="-linear ",
        desc="fill in values with linear interpolation along strongest gradient",
    )

    exponent = traits.Float(
        argstr="-exponent %f ",
        position=9,
        default=6.0,
        desc="exponent n to use in (area / (distance ^ n)) as the "
        "weighting function (default 6)",
    )

    corrected_areas = File(
        argstr="-corrected-areas %s ",
        position=10,
        desc="vertex areas to use instead of computing them from the surface",
    )

    legacy_cutoff = traits.Bool(
        position=11,
        argstr="-legacy-cutoff ",
        desc="use the v1.3.2 method of choosing how many vertices to "
        "use when calculating the dilated value with weighted method",
    )


class MetricDilateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output file")


class MetricDilate(WBCommand):
    """
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
    _cmd = "wb_command -metric-dilate "


class MetricResampleInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The metric file to resample",
    )
    current_sphere = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="A sphere surface with the mesh that the metric is currently on",
    )
    new_sphere = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=2,
        desc="A sphere surface that is in register with <current-sphere> and"
        " has the desired output mesh",
    )
    method = traits.Enum(
        "ADAP_BARY_AREA",
        "BARYCENTRIC",
        argstr="%s",
        mandatory=True,
        position=3,
        desc="The method name - ADAP_BARY_AREA method is recommended for"
        " ordinary metric data, because it should use all data while"
        " downsampling, unlike BARYCENTRIC. If ADAP_BARY_AREA is used,"
        " exactly one of area_surfs or area_metrics must be specified",
    )
    out_file = File(
        name_source=["new_sphere"],
        name_template="%s.out",
        keep_extension=True,
        argstr="%s",
        position=4,
        desc="The output metric",
    )
    area_surfs = traits.Bool(
        position=5,
        argstr="-area-surfs",
        xor=["area_metrics"],
        desc="Specify surfaces to do vertex area correction based on",
    )
    area_metrics = traits.Bool(
        position=5,
        argstr="-area-metrics",
        xor=["area_surfs"],
        desc="Specify vertex area metrics to do area correction based on",
    )
    current_area = File(
        exists=True,
        position=6,
        argstr="%s",
        desc="A relevant anatomical surface with <current-sphere> mesh OR"
        " a metric file with vertex areas for <current-sphere> mesh",
    )
    new_area = File(
        exists=True,
        position=7,
        argstr="%s",
        desc="A relevant anatomical surface with <current-sphere> mesh OR"
        " a metric file with vertex areas for <current-sphere> mesh",
    )
    roi_metric = File(
        exists=True,
        position=8,
        argstr="-current-roi %s",
        desc="Input roi on the current mesh used to exclude non-data vertices",
    )
    valid_roi_out = traits.Bool(
        position=9,
        argstr="-valid-roi-out",
        desc="Output the ROI of vertices that got data from valid source vertices",
    )
    largest = traits.Bool(
        position=10,
        argstr="-largest",
        desc="Use only the value of the vertex with the largest weight",
    )


class MetricResampleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output metric")
    roi_file = File(desc="ROI of vertices that got data from valid source vertices")


class MetricResample(WBCommand):
    """
    Resample a metric file to a different mesh

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
    _cmd = "wb_command -metric-resample"

    def _format_arg(self, opt, spec, val):
        if opt in ["current_area", "new_area"]:
            if not self.inputs.area_surfs and not self.inputs.area_metrics:
                raise ValueError(
                    "{} was set but neither area_surfs or" " area_metrics were set".format(opt)
                )
        if opt == "method":
            if (
                val == "ADAP_BARY_AREA"
                and not self.inputs.area_surfs
                and not self.inputs.area_metrics
            ):
                raise ValueError("Exactly one of area_surfs or area_metrics" " must be specified")
        if opt == "valid_roi_out" and val:
            # generate a filename and add it to argstr
            roi_out = self._gen_filename(self.inputs.in_file, suffix="_roi")
            iflogger.info("Setting roi output file as", roi_out)
            spec.argstr += " " + roi_out
        return super(MetricResample, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = super(MetricResample, self)._list_outputs()
        if self.inputs.valid_roi_out:
            roi_file = self._gen_filename(self.inputs.in_file, suffix="_roi")
            outputs["roi_file"] = os.path.abspath(roi_file)
        return outputs
