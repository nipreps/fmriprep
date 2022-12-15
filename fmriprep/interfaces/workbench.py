import os

from nipype.interfaces.base import CommandLineInputSpec, File, Str, TraitedSpec, traits
from nipype.interfaces.base.traits_extension import (
    InputMultiObject,
    OutputMultiObject,
    isdefined,
)
from nipype.interfaces.workbench.base import WBCommand

# patch
from nipype.interfaces.workbench.cifti import CiftiSmooth as _CiftiSmooth
from nipype.interfaces.workbench.cifti import (
    CiftiSmoothInputSpec as _CiftiSmoothInputSpec,
)

VALID_STRUCTURES = (
    "CORTEX_LEFT",
    "CORTEX_RIGHT",
    "CEREBELLUM",
    "ACCUMBENS_LEFT",
    "ACCUMBENS_RIGHT",
    "ALL_GREY_MATTER",
    "ALL_WHITE_MATTER",
    "AMYGDALA_LEFT",
    "AMYGDALA_RIGHT",
    "BRAIN_STEM",
    "CAUDATE_LEFT",
    "CAUDATE_RIGHT",
    "CEREBELLAR_WHITE_MATTER_LEFT",
    "CEREBELLAR_WHITE_MATTER_RIGHT",
    "CEREBELLUM_LEFT",
    "CEREBELLUM_RIGHT",
    "CEREBRAL_WHITE_MATTER_LEFT",
    "CEREBRAL_WHITE_MATTER_RIGHT",
    "CORTEX",
    "DIENCEPHALON_VENTRAL_LEFT",
    "DIENCEPHALON_VENTRAL_RIGHT",
    "HIPPOCAMPUS_LEFT",
    "HIPPOCAMPUS_RIGHT",
    "INVALID",
    "OTHER",
    "OTHER_GREY_MATTER",
    "OTHER_WHITE_MATTER",
    "PALLIDUM_LEFT",
    "PALLIDUM_RIGHT",
    "PUTAMEN_LEFT",
    "PUTAMEN_RIGHT",
    "THALAMUS_LEFT",
    "THALAMUS_RIGHT",
)


class CiftiCreateDenseFromTemplateInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="File to match brainordinates of",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="template_%s.nii",
        keep_extension=True,
        argstr="%s",
        position=1,
        desc="The output CIFTI file",
    )
    series = traits.Bool(
        argstr="-series",
        position=2,
        desc="Make a dtseries file instead of a dscalar",
    )
    series_step = traits.Float(
        requires=["series"],
        argstr="%.1f",
        position=3,
        desc="Increment between series points",
    )
    series_start = traits.Float(
        requires=["series"],
        argstr="%.1f",
        position=4,
        desc="Start value of the series",
    )
    series_unit = traits.Enum(
        "SECOND",
        "HERTZ",
        "METER",
        "RADIAN",
        requries=["series"],
        argstr="-unit %s",
        position=5,
        desc="select unit for series (default SECOND)",
    )
    volume_all = traits.File(
        exists=True,
        argstr="-volume-all %s",
        position=6,
        desc="the input volume file for all voxel data",
    )
    volume_all_from_cropped = traits.Bool(
        requires=["volume_all"],
        argstr="-from-cropped",
        position=7,
        desc="the input is cropped to the size of the voxel data in the template file",
    )
    label_collision = traits.Enum(
        "ERROR",
        "SURFACES_FIRST",
        "LEGACY",
        argstr="-label-collision %s",
        position=8,
        desc="how to handle conflicts between label keys, use 'LEGACY' to match v1.4.2 "
        "and earlier",
    )
    cifti = InputMultiObject(
        File(exists=True),
        argstr="-cifti %s",
        position=9,
        desc="use input data from one or more CIFTI files",
    )
    metric = InputMultiObject(
        traits.Tuple(traits.Enum(VALID_STRUCTURES), File(exists=True)),
        argstr="%s",
        position=10,
        desc="use input data from one or more metric files",
    )
    label = InputMultiObject(
        traits.Tuple(traits.Enum(VALID_STRUCTURES), File(exists=True)),
        argstr="%s",
        position=11,
        desc="use input data from one or more surface label files",
    )
    volume = InputMultiObject(
        traits.Either(
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File(exists=True)),
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File(exists=True), traits.Bool()),
        ),
        argstr="%s",
        position=12,
        desc="use a volume file for a single volume structure's data",
    )


class CiftiCreateDenseFromTemplateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="The output CIFTI file")


class CiftiCreateDenseFromTemplate(WBCommand):
    """
    This command helps you make a new dscalar, dtseries, or dlabel cifti file
    that matches the brainordinate space used in another cifti file.  The
    template file must have the desired brainordinate space in the mapping
    along the column direction (for dtseries, dscalar, dlabel, and symmetric
    dconn this is always the case).  All input cifti files must have a brain
    models mapping along column and use the same volume space and/or surface
    vertex count as the template for structures that they contain.  If any
    input files contain label data, then input files with non-label data are
    not allowed, and the -series option may not be used.

    Any structure that isn't covered by an input is filled with zeros or the
    unlabeled key.

    >>> from nibabies.interfaces import workbench as wb
    >>> frmtpl = wb.CiftiCreateDenseFromTemplate()
    >>> frmtpl.inputs.in_file = data_dir / "func.dtseries.nii"
    >>> frmtpl.inputs.series = True
    >>> frmtpl.inputs.series_step = 0.8
    >>> frmtpl.inputs.series_start = 0
    >>> frmtpl.cmdline
    'wb_command -cifti-create-dense-from-template .../func.dtseries.nii \
    template_func.dtseries.nii -series 0.8 0.0'

    >>> frmtpl.inputs.volume = [("OTHER", data_dir / 'functional.nii', True), \
        ("PUTAMEN_LEFT", data_dir / 'functional.nii')]
    >>> frmtpl.cmdline
    'wb_command -cifti-create-dense-from-template .../func.dtseries.nii \
    template_func.dtseries.nii -series 0.8 0.0 \
    -volume OTHER .../functional.nii -from-cropped \
    -volume PUTAMEN_LEFT .../functional.nii'
    """

    input_spec = CiftiCreateDenseFromTemplateInputSpec
    output_spec = CiftiCreateDenseFromTemplateOutputSpec
    _cmd = "wb_command -cifti-create-dense-from-template"

    def _format_arg(self, name, trait_spec, value):
        if name in ("metric", "label", "volume"):
            cmds = []
            for val in value:
                if val[-1] is True:  # volume specific
                    val = val[:2] + ("-from-cropped",)
                cmds.append(" ".join((f"-{name}",) + val))
            return trait_spec.argstr % " ".join(cmds)
        return super()._format_arg(name, trait_spec, value)


class CiftiCreateDenseTimeseriesInputSpec(CommandLineInputSpec):
    out_file = File(
        value="out.dtseries.nii",
        usedefault=True,
        argstr="%s",
        position=0,
        desc="The output CIFTI file",
    )
    volume_data = File(
        exists=True,
        argstr="-volume %s",
        position=1,
        requires=["volume_structure_labels"],
        desc="volume file containing all voxel data for all volume structures",
    )
    volume_structure_labels = File(
        exists=True,
        argstr="%s",
        position=2,
        requires=["volume_data"],
        desc="label volume file containing labels for cifti structures",
    )
    left_metric = File(
        exists=True,
        argstr="-left-metric %s",
        position=3,
        desc="metric file for left surface",
    )
    roi_left = File(
        exists=True,
        argstr="-roi-left %s",
        position=4,
        requires=["left_metric"],
        desc="ROI (as metric file) of vertices to use from left surface",
    )
    right_metric = File(
        exists=True,
        argstr="-right-metric %s",
        position=5,
        desc="metric file for right surface",
    )
    roi_right = File(
        exists=True,
        argstr="-roi-right %s",
        position=6,
        requires=["right_metric"],
        desc="ROI (as metric file) of vertices to use from right surface",
    )
    cerebellum_metric = File(
        exists=True,
        argstr="-cerebellum-metric %s",
        position=7,
        desc="metric file for cerebellum",
    )
    roi_cerebellum = File(
        exists=True,
        argstr="-roi-cerebellum %s",
        position=8,
        requires=["cerebellum_metric"],
        desc="ROI (as metric file) of vertices to use from cerebellum",
    )
    timestart = traits.Float(
        argstr="-timestart %g",
        position=9,
        desc="the time at the first frame, in seconds",
    )
    timestep = traits.Float(
        argstr="-timestep %g",
        position=10,
        desc="the timestep, in seconds",
    )
    unit = traits.Enum(
        "SECOND",
        "HERTZ",
        "METER",
        "RADIAN",
        argstr="-unit %s",
        position=11,
        desc="use a unit other than time",
    )


class CiftiCreateDenseTimeseriesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="CIFTI dense timeseries file")


class CiftiCreateDenseTimeseries(WBCommand):
    """
    Create a CIFTI dense timeseries.

    All input files must have the same number of columns/subvolumes.  Only
    the specified components will be in the output cifti.  At least one
    component must be specified.

    See -volume-label-import and -volume-help for format details of label
    volume files.  The structure-label-volume should have some of the label
    names from this list, all other label names will be ignored:

    CORTEX_LEFT
    CORTEX_RIGHT
    CEREBELLUM
    ACCUMBENS_LEFT
    ACCUMBENS_RIGHT
    ALL_GREY_MATTER
    ALL_WHITE_MATTER
    AMYGDALA_LEFT
    AMYGDALA_RIGHT
    BRAIN_STEM
    CAUDATE_LEFT
    CAUDATE_RIGHT
    CEREBELLAR_WHITE_MATTER_LEFT
    CEREBELLAR_WHITE_MATTER_RIGHT
    CEREBELLUM_LEFT
    CEREBELLUM_RIGHT
    CEREBRAL_WHITE_MATTER_LEFT
    CEREBRAL_WHITE_MATTER_RIGHT
    CORTEX
    DIENCEPHALON_VENTRAL_LEFT
    DIENCEPHALON_VENTRAL_RIGHT
    HIPPOCAMPUS_LEFT
    HIPPOCAMPUS_RIGHT
    INVALID
    OTHER
    OTHER_GREY_MATTER
    OTHER_WHITE_MATTER
    PALLIDUM_LEFT
    PALLIDUM_RIGHT
    PUTAMEN_LEFT
    PUTAMEN_RIGHT
    THALAMUS_LEFT
    THALAMUS_RIGHT

    >>> from nibabies.interfaces.workbench import CiftiCreateDenseTimeseries
    >>> createdts = CiftiCreateDenseTimeseries()
    >>> createdts.inputs.volume_data = data_dir /'functional.nii'
    >>> createdts.inputs.volume_structure_labels = data_dir / 'atlas.nii'
    >>> createdts.cmdline
    'wb_command -cifti-create-dense-timeseries out.dtseries.nii \
    -volume .../functional.nii .../atlas.nii'
    """

    input_spec = CiftiCreateDenseTimeseriesInputSpec
    output_spec = CiftiCreateDenseTimeseriesOutputSpec
    _cmd = "wb_command -cifti-create-dense-timeseries"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs


class CiftiCreateLabelInputSpec(CommandLineInputSpec):
    out_file = File(
        value="out.dlabel.nii",
        usedefault=True,
        argstr="%s",
        position=0,
        desc="the output CIFTI file",
    )
    volume_label = File(
        exists=True,
        requires=["structure_label_volume"],
        argstr="-volume %s",
        position=1,
        desc="label volume file containing the data to be copied",
    )
    structure_label_volume = File(
        exists=True,
        requires=["volume_label"],
        argstr="%s",
        position=2,
        desc="label volume file that defines which voxels to use",
    )
    left_label = File(
        exists=True,
        argstr="-left-label %s",
        position=3,
        desc="Label file for left surface",
    )
    left_roi = File(
        exists=True,
        requires=["left_label"],
        argstr="-roi-left %s",
        position=4,
        desc="roi of vertices to use from left surface as a metric file",
    )
    right_label = File(
        exists=True,
        argstr="-right-label %s",
        position=5,
        desc="Label file for right surface",
    )
    right_roi = File(
        exists=True,
        requires=["right_label"],
        argstr="-roi-right %s",
        position=6,
        desc="roi of vertices to use from right surface as a metric file",
    )
    cerebellum_label = File(
        exists=True,
        argstr="-cerebellum-label %s",
        position=7,
        desc="label for the cerebellum",
    )
    cerebellum_roi = File(
        exists=True,
        requires=["cerebellum_label"],
        argstr="-roi-cerebellum %s",
        position=8,
        desc="roi of vertices to use from cerebellum",
    )


class CiftiCreateLabelOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output CIFTI file")


class CiftiCreateLabel(WBCommand):
    """
    All input files must have the same number of columns/subvolumes.  Only
    the specified components will be in the output cifti.  At least one
    component must be specified.

    The -volume option requires two volume arguments, the label-volume
    argument contains all labels you want to display (e.g. nuclei of the
    thalamus), whereas the structure-label-volume argument contains all CIFTI
    voxel-based structures you want to include data within (e.g.
    THALAMUS_LEFT, THALAMUS_RIGHT, etc).  See -volume-label-import and
    -volume-help for format details of label volume files.  If you just want
    the labels in voxels to be the structure names, you may use the same file
    for both arguments.  The structure-label-volume must use some of the
    label names from this list, all other label names in the
    structure-label-volume will be ignored:

    CORTEX_LEFT
    CORTEX_RIGHT
    CEREBELLUM
    ACCUMBENS_LEFT
    ACCUMBENS_RIGHT
    ALL_GREY_MATTER
    ALL_WHITE_MATTER
    AMYGDALA_LEFT
    AMYGDALA_RIGHT
    BRAIN_STEM
    CAUDATE_LEFT
    CAUDATE_RIGHT
    CEREBELLAR_WHITE_MATTER_LEFT
    CEREBELLAR_WHITE_MATTER_RIGHT
    CEREBELLUM_LEFT
    CEREBELLUM_RIGHT
    CEREBRAL_WHITE_MATTER_LEFT
    CEREBRAL_WHITE_MATTER_RIGHT
    CORTEX
    DIENCEPHALON_VENTRAL_LEFT
    DIENCEPHALON_VENTRAL_RIGHT
    HIPPOCAMPUS_LEFT
    HIPPOCAMPUS_RIGHT
    INVALID
    OTHER
    OTHER_GREY_MATTER
    OTHER_WHITE_MATTER
    PALLIDUM_LEFT
    PALLIDUM_RIGHT
    PUTAMEN_LEFT
    PUTAMEN_RIGHT
    THALAMUS_LEFT
    THALAMUS_RIGHT

    >>> from nibabies.interfaces import workbench as wb
    >>> lab = wb.CiftiCreateLabel()
    >>> lab.inputs.volume_label = data_dir / "functional.nii"
    >>> lab.inputs.structure_label_volume = data_dir / "functional.nii"
    >>> lab.cmdline
    'wb_command -cifti-create-label out.dlabel.nii -volume .../functional.nii .../functional.nii'
    """

    input_spec = CiftiCreateLabelInputSpec
    output_spec = CiftiCreateLabelOutputSpec
    _cmd = "wb_command -cifti-create-label"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs


class CiftiDilateInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input CIFTI file",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s",
        position=1,
        desc="Which dimension to dilate along, ROW or COLUMN",
    )
    surface_distance = traits.Int(
        mandatory=True,
        argstr="%d",
        position=2,
        desc="The distance to dilate on surfaces, in mm",
    )
    volume_distance = traits.Int(
        mandatory=True,
        argstr="%d",
        position=3,
        desc="The distance to dilate in the volume, in mm",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="dilated_%s.nii",
        keep_extension=True,
        argstr="%s",
        position=4,
        desc="The dilated CIFTI file",
    )
    left_surface = File(
        exists=True,
        position=5,
        argstr="-left-surface %s",
        desc="Specify the left surface to use",
    )
    left_corrected_areas = File(
        exists=True,
        position=6,
        requires=["left_surface"],
        argstr="-left-corrected-areas %s",
        desc="vertex areas (as a metric) to use instead of computing them from the left surface.",
    )
    right_surface = File(
        exists=True,
        position=7,
        argstr="-right-surface %s",
        desc="Specify the right surface to use",
    )
    right_corrected_areas = File(
        exists=True,
        position=8,
        requires=["right_surface"],
        argstr="-right-corrected-areas %s",
        desc="vertex areas (as a metric) to use instead of computing them from the right surface",
    )
    cerebellum_surface = File(
        exists=True,
        position=9,
        argstr="-cerebellum-surface %s",
        desc="specify the cerebellum surface to use",
    )
    cerebellum_corrected_areas = File(
        exists=True,
        position=10,
        requires=["cerebellum_surface"],
        argstr="-cerebellum-corrected-areas %s",
        desc="vertex areas (as a metric) to use instead of computing them from the cerebellum "
        "surface",
    )
    bad_brainordinate_roi = File(
        exists=True,
        position=11,
        argstr="-bad-brainordinate-roi %s",
        desc="CIFTI dscalar or dtseries file, positive values denote brainordinates to have their "
        "values replaced",
    )
    nearest = traits.Bool(
        position=12,
        argstr="-nearest",
        desc="Use nearest good value instead of a weighted average",
    )
    merged_volume = traits.Bool(
        position=13,
        argstr="-merged-volume",
        desc="treat volume components as if they were a single component",
    )
    legacy_mode = traits.Bool(
        position=14,
        argstr="-legacy-mode",
        desc="Use the math from v1.3.2 and earlier for weighted dilation",
    )


class CiftiDilateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Dilated CIFTI file")


class CiftiDilate(WBCommand):
    """
    Dilate a CIFTI file.

    For all data values designated as bad, if they neighbor a good value or
    are within the specified distance of a good value in the same kind of
    model, replace the value with a distance weighted average of nearby good
    values, otherwise set the value to zero.  If -nearest is specified, it
    will use the value from the closest good value within range instead of a
    weighted average.  When the input file contains label data, nearest
    dilation is used on the surface, and weighted popularity is used in the
    volume.

    The -*-corrected-areas options are intended for dilating on group average
    surfaces, but it is only an approximate correction for the reduction of
    structure in a group average surface.

    If -bad-brainordinate-roi is specified, all values, including those with
    value zero, are good, except for locations with a positive value in the
    ROI.  If it is not specified, only values equal to zero are bad.

    """

    input_spec = CiftiDilateInputSpec
    output_spec = CiftiDilateOutputSpec
    _cmd = "wb_command -cifti-dilate"


class CiftiResampleInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="the CIFTI file to resample",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s",
        position=1,
        desc="the direction of the input that should be resampled",
    )
    template = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=2,
        desc="a CIFTI file containing the CIFTI space to resample to",
    )
    template_direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s",
        position=3,
        desc="the direction of the template to use as the resampling space",
    )
    surface_method = traits.Enum(
        "ADAP_BARY_AREA",
        "BARYCENTRIC",
        mandatory=True,
        argstr="%s",
        position=4,
        desc="surface resampling method",
    )
    volume_method = traits.Enum(
        "CUBIC",
        "ENCLOSING_VOXEL",
        "TRILINEAR",
        mandatory=True,
        argstr="%s",
        position=5,
        desc="volume interpolation method",
    )
    out_file = File(
        name_source=["in_file"],
        keep_extension=True,
        name_template="resampled_%s.nii",
        argstr="%s",
        position=6,
        desc="the output cifti file",
    )
    surface_largest = traits.Bool(
        argstr="-surface-largest",
        position=7,
        desc="use largest weight instead of weighted average or popularity when doing "
        "surface resampling",
    )
    volume_predilate = traits.Int(
        argstr="-volume-predilate %d",
        position=8,
        desc="distance, in mm, to dilate the volume components before resampling",
    )
    volume_predilate_nearest = traits.Bool(
        requires=["volume_predilate"],
        xor=["volume_predilate_weighted"],
        argstr="-nearest",
        position=9,
        desc="use nearest value dilation",
    )
    volume_predilate_weighted = traits.Bool(
        requires=["volume_predilate"],
        xor=["volume_predilate_nearest"],
        argstr="-weighted",
        position=10,
        desc="use weighted dilation (default)",
    )
    volume_predilate_weighted_exponent = traits.Int(
        requires=["volume_predilate_weighted"],
        argstr="-exponent %d",
        position=11,
        desc="exponent 'n' to use in (1 / (distance ^ n)) as the weighting function (default 7)",
    )
    volume_predilate_weighted_legacy = traits.Bool(
        requires=["volume_predilate_weighted"],
        argstr="-legacy-cutoff",
        position=12,
        desc="use v1.3.2 logic for the kernel cutoff",
    )
    surface_postdilate = traits.Int(
        argstr="-surface-postdilate %d",
        position=13,
        desc="distance, in mm, to dilate the surface components after resampling",
    )
    surface_postdilate_nearest = traits.Bool(
        requires=["surface_postdilate"],
        xor=["surface_postdilate_weighted", "surface_postdilate_linear"],
        argstr="-nearest",
        position=14,
        desc="use nearest value dilation",
    )
    surface_postdilate_linear = traits.Bool(
        requires=["surface_postdilate"],
        xor=["surface_postdilate_weighted", "surface_postdilate_nearest"],
        argstr="-linear",
        position=15,
        desc="use linear dilation",
    )
    surface_postdilate_weighted = traits.Bool(
        requires=["surface_postdilate"],
        xor=["surface_postdilate_nearest", "surface_postdilate_linear"],
        argstr="-weighted",
        position=16,
        desc="use weighted dilation (default for non-label data)",
    )
    surface_postdilate_weighted_exponent = traits.Int(
        requires=["surface_postdilate_weighted"],
        argstr="-exponent %d",
        position=17,
        desc="exponent 'n' to use in (area / (distance ^ n)) as the weighting function "
        "(default 6)",
    )
    surface_postdilate_weighted_legacy = traits.Bool(
        requires=["surface_postdilate_weighted"],
        argstr="-legacy-cutoff",
        position=18,
        desc="use v1.3.2 logic for the kernel cutoff",
    )
    affine = File(
        exists=True,
        argstr="-affine %s",
        position=19,
        desc="affine file for transformation on the volume components",
    )
    affine_flirt_source = File(
        exists=True,
        requires=["affine", "affine_flirt_target"],
        argstr="-flirt %s",
        position=20,
        desc="the source volume used when generating the affine. MUST be used if affine is "
        "a flirt affine",
    )
    affine_flirt_target = File(
        exists=True,
        requires=["affine", "affine_flirt_source"],
        argstr="%s",
        position=21,
        desc="the target volume used when generating the affine. MUST be used if affine is "
        "a flirt affine",
    )
    warpfield = File(
        exists=True,
        argstr="-warpfield %s",
        position=22,
        desc="the warpfield to use on the volume components",
    )
    warpfield_fnirt_source = File(
        exists=True,
        requires=["warpfield"],
        argstr="-fnirt %s",
        position=23,
        desc="the source volume used when generating the warpfield. MUST be used if using "
        "a fnirt warpfield",
    )
    left_sphere_current = File(
        exists=True,
        requires=["left_sphere_new"],
        argstr="-left-spheres %s",
        position=24,
        desc="a sphere with the same mesh as the current left surface",
    )
    left_sphere_new = File(
        exists=True,
        requires=["left_sphere_current"],
        argstr="%s",
        position=25,
        desc="a sphere with the new left mesh that is in register with the current sphere",
    )
    left_area_surf_current = File(
        exists=True,
        requires=["left_sphere_current", "left_area_surf_new"],
        argstr="-left-area-surfs %s",
        position=26,
        desc="a relevant left anatomical surface with current mesh",
    )
    left_area_surf_new = File(
        exists=True,
        requires=["left_sphere_new", "left_area_surf_current"],
        argstr="%s",
        position=27,
        desc="a relevant left anatomical surface with new mesh",
    )
    left_area_metric_current = File(
        exists=True,
        requires=["left_sphere_current", "left_area_metric_new"],
        argstr="-left-area-metrics %s",
        position=28,
        desc="a metric file with vertex areas for the current mesh",
    )
    left_area_metric_new = File(
        exists=True,
        requires=["left_sphere_new", "left_area_metric_current"],
        argstr="%s",
        position=29,
        desc="a metric file with vertex areas for the new mesh",
    )
    right_sphere_current = File(
        exists=True,
        requires=["right_sphere_new"],
        argstr="-right-spheres %s",
        position=30,
        desc="a sphere with the same mesh as the current right surface",
    )
    right_sphere_new = File(
        exists=True,
        requires=["right_sphere_current"],
        argstr="%s",
        position=31,
        desc="a sphere with the new right mesh that is in register with the current sphere",
    )
    right_area_surf_current = File(
        exists=True,
        requires=["right_sphere_current", "right_area_surf_new"],
        argstr="-right-area-surfs %s",
        position=32,
        desc="a relevant right anatomical surface with current mesh",
    )
    right_area_surf_new = File(
        exists=True,
        requires=["right_sphere_new", "right_area_surf_current"],
        argstr="%s",
        position=33,
        desc="a relevant right anatomical surface with new mesh",
    )
    right_area_metric_current = File(
        exists=True,
        requires=["right_sphere_current", "right_area_metric_new"],
        argstr="-right-area-metrics %s",
        position=34,
        desc="a metric file with vertex areas for the current mesh",
    )
    right_area_metric_new = File(
        exists=True,
        requires=["right_sphere_new", "right_area_metric_current"],
        argstr="%s",
        position=35,
        desc="a metric file with vertex areas for the new mesh",
    )
    cerebellum_sphere_current = File(
        exists=True,
        requires=["cerebellum_sphere_new"],
        argstr="-cerebellum-spheres %s",
        position=36,
        desc="a sphere with the same mesh as the current cerebellum surface",
    )
    cerebellum_sphere_new = File(
        exists=True,
        requires=["cerebellum_sphere_current"],
        argstr="%s",
        position=37,
        desc="a sphere with the new cerebellum mesh that is in register with the current sphere",
    )
    cerebellum_area_surf_current = File(
        exists=True,
        requires=["cerebellum_sphere_current", "cerebellum_area_surf_new"],
        argstr="-cerebellum-area-surfs %s",
        position=38,
        desc="a relevant cerebellum anatomical surface with current mesh",
    )
    cerebellum_area_surf_new = File(
        exists=True,
        requires=["cerebellum_sphere_new", "cerebellum_area_surf_current"],
        argstr="%s",
        position=39,
        desc="a relevant cerebellum anatomical surface with new mesh",
    )
    cerebellum_area_metric_current = File(
        exists=True,
        requires=["cerebellum_sphere_current", "cerebellum_area_metric_new"],
        argstr="-cerebellum-area-metrics %s",
        position=40,
        desc="a metric file with vertex areas for the current mesh",
    )
    cerebellum_area_metric_new = File(
        exists=True,
        requires=["cerebellum_sphere_new", "cerebellum_area_metric_current"],
        argstr="%s",
        position=41,
        desc="a metric file with vertex areas for the new mesh",
    )


class CiftiResampleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the resampled CIFTI file")


class CiftiResample(WBCommand):
    """
    Resample cifti data to a different brainordinate space.  Use COLUMN for
    the direction to resample dscalar, dlabel, or dtseries.  Resampling both
    dimensions of a dconn requires running this command twice, once with
    COLUMN and once with ROW.  If you are resampling a dconn and your machine
    has a large amount of memory, you might consider using
    -cifti-resample-dconn-memory to avoid writing and rereading an
    intermediate file.  The <template-direction> argument should usually be
    COLUMN, as dtseries, dscalar, and dlabel all have brainordinates on that
    direction.  If spheres are not specified for a surface structure which
    exists in the cifti files, its data is copied without resampling or
    dilation.  Dilation is done with the 'nearest' method, and is done on
    <new-sphere> for surface data.  Volume components are padded before
    dilation so that dilation doesn't run into the edge of the component
    bounding box.  If neither -affine nor -warpfield are specified, the
    identity transform is assumed for the volume data.

    The recommended resampling methods are ADAP_BARY_AREA and CUBIC (cubic
    spline), except for label data which should use ADAP_BARY_AREA and
    ENCLOSING_VOXEL.  Using ADAP_BARY_AREA requires specifying an area option
    to each used -*-spheres option.

    >>> from nibabies.interfaces import workbench as wb
    >>> res = wb.CiftiResample()
    >>> res.inputs.in_file = data_dir / "func.dtseries.nii"
    >>> res.inputs.direction = "COLUMN"
    >>> res.inputs.template = data_dir / "func.dlabel.nii"
    >>> res.inputs.template_direction = "COLUMN"
    >>> res.inputs.surface_method = "ADAP_BARY_AREA"
    >>> res.inputs.volume_method = "CUBIC"
    >>> res.inputs.out_file = "resampled.dtseries.nii"
    >>> res.inputs.volume_predilate = 10
    >>> res.cmdline
    'wb_command -cifti-resample .../func.dtseries.nii COLUMN .../func.dlabel.nii COLUMN \
    ADAP_BARY_AREA CUBIC resampled.dtseries.nii -volume-predilate 10'
    """

    input_spec = CiftiResampleInputSpec
    output_spec = CiftiResampleOutputSpec
    _cmd = "wb_command -cifti-resample"


class CiftiSeparateInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="the cifti to separate a component of",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s",
        position=1,
        desc="which dimension to smooth along, ROW or COLUMN",
    )
    volume_all_file = File(
        argstr="-volume-all %s",
        position=2,
        desc="separate all volume structures into a volume file",
    )
    volume_all_roi_file = File(
        argstr="-roi %s",
        position=3,
        requires=["volume_all_file"],
        desc="output the roi of which voxels have data",
    )
    volume_all_label_file = File(
        argstr="-label %s",
        position=4,
        requires=["volume_all_file"],
        desc="output a volume label file indicating the location of structures",
    )
    volume_all_crop = traits.Bool(
        argstr="-crop",
        position=5,
        requires=["volume_all_file"],
        desc="crop volume to the size of the data rather than using the original volume size",
    )
    # the following can be repeated
    label = InputMultiObject(
        traits.Either(
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File()),
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File(), File()),
        ),
        argstr="%s",
        position=6,
        desc="separate one or more surface models into a surface label file",
    )
    metric = InputMultiObject(
        traits.Either(
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File()),
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File(), File()),  # -roi
        ),
        argstr="%s",
        position=7,
        desc="separate one or more surface models into a metric file",
    )
    volume = InputMultiObject(
        traits.Either(
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File()),
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File(), File()),  # -roi
            traits.Tuple(traits.Enum(VALID_STRUCTURES, File(), traits.Bool)),  # -crop
            traits.Tuple(traits.Enum(VALID_STRUCTURES), File(), File(), traits.Bool),  # -roi -crop
        ),
        argstr="%s",
        position=8,
        desc="separate one or more volume structures into a volume file",
    )


class CiftiSeparateOutputSpec(TraitedSpec):
    volume_all_file = File(desc="File containing all volume structures")
    volume_all_roi_file = File(desc="Output the roi of which voxels have data")
    volume_all_label_file = File(
        desc="output a volume label file indicating the location of structures"
    )
    label_files = OutputMultiObject(File(), desc="Output label files")
    label_roi_files = OutputMultiObject(File(), desc="Output label rois files")
    metric_files = OutputMultiObject(File(), desc="Output metric files")
    metric_roi_files = OutputMultiObject(File(), desc="Output metric rois files")
    volume_files = OutputMultiObject(File(), desc="Output volume files")
    volume_roi_files = OutputMultiObject(File(), desc="Output volume roi files")


class CiftiSeparate(WBCommand):
    """
    Extract left or right hemisphere surface from CIFTI file (.dtseries)
    other structure can also be extracted
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries.

    >>> separate = CiftiSeparate()
    >>> separate.inputs.in_file = data_dir / "func.dtseries.nii"
    >>> separate.inputs.direction = "COLUMN"
    >>> separate.inputs.volume_all_file = "volume_all.nii.gz"
    >>> separate.cmdline
    'wb_command -cifti-separate .../func.dtseries.nii COLUMN \
    -volume-all volume_all.nii.gz'

    Metrics, labels, and volumes can also be freely extracted
    >>> separate.inputs.metric = [("CORTEX_LEFT", "cortexleft.func.gii")]
    >>> separate.inputs.volume = [("HIPPOCAMPUS_LEFT", "hippoL.nii.gz"), \
        ("HIPPOCAMPUS_RIGHT", "hippoR.nii.gz", "hippoR.roi.nii.gz")]
    >>> separate.cmdline
    'wb_command -cifti-separate .../func.dtseries.nii COLUMN \
    -volume-all volume_all.nii.gz -metric CORTEX_LEFT cortexleft.func.gii \
    -volume HIPPOCAMPUS_LEFT hippoL.nii.gz \
    -volume HIPPOCAMPUS_RIGHT hippoR.nii.gz -roi hippoR.roi.nii.gz'

    """

    input_spec = CiftiSeparateInputSpec
    output_spec = CiftiSeparateOutputSpec
    _cmd = "wb_command -cifti-separate"
    _label_roi_files = []
    _metric_roi_files = []
    _volume_roi_files = []

    def _format_arg(self, name, trait_spec, value):
        if name in ("label", "metric", "volume"):
            cmds = []
            for i, val in enumerate(value):
                if len(val) == 3:
                    if val[-1] is True:
                        val = val[:-1] + ("-crop",)
                    else:
                        val = val[:-1] + ("-roi", val[-1])
                        self._set_roi_file(name, val[-1])
                elif len(val) == 4:
                    val = val[:-2] + ("-roi", val[-2]) + ("crop") if val[-1] is True else ()
                    self._set_roi_file(name, val[-2])
                cmds.append(" ".join((f"-{name}",) + val))
            return trait_spec.argstr % " ".join(cmds)
        return super()._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.volume_all_file:
            outputs["volume_all_file"] = os.path.abspath(self.inputs.volume_all_file)
        if self.inputs.volume_all_roi_file:
            outputs["volume_all_roi_file"] = os.path.abspath(self.inputs.volume_all_roi_file)
        if self.inputs.volume_all_label_file:
            outputs["volume_all_label_file"] = os.path.abspath(self.inputs.volume_all_label_file)
        if self.inputs.label:
            for label in self.inputs.label:
                outputs["label_files"] = (outputs["label_files"] or []) + self._gen_filename(
                    label[2]
                )
            if self._label_roi_files:
                outputs["label_roi_files"] = self._label_roi_files
        if self.inputs.metric:
            for metric in self.inputs.metric:
                outputs["metric_files"] = (outputs["metric_files"] or []) + self._gen_filename(
                    metric[2]
                )
            if self._metric_roi_files:
                outputs["metric_roi_files"] = self._metric_roi_files
        if self.inputs.volume:
            for volume in self.inputs.volume:
                outputs["volume_files"] = (outputs["volume_files"] or []) + self._gen_filename(
                    volume[2]
                )
            if self._volume_roi_files:
                outputs["volume_roi_files"] = self._volume_roi_files
        return outputs

    def _set_roi_file(self, name, file):
        rois = getattr(self, f"_{name}_roi_files")
        rois.append(self._gen_filename(file))


class CiftiSmoothInputSpec(_CiftiSmoothInputSpec):
    left_surf = File(
        exists=True,
        position=5,
        argstr="-left-surface %s",
        desc="Specify the left surface to use",
    )
    right_surf = File(
        exists=True,
        position=7,
        argstr="-right-surface %s",
        desc="Specify the right surface to use",
    )


class CiftiSmooth(_CiftiSmooth):
    input_spec = CiftiSmoothInputSpec


class VolumeAffineResampleInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="volume to resample",
    )
    volume_space = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="a volume file in the volume space you want for the output",
    )
    method = traits.Enum(
        "CUBIC",
        "ENCLOSING_VOXEL",
        "TRILINEAR",
        mandatory=True,
        argstr="%s",
        position=2,
        desc="The resampling method. The recommended methods are CUBIC "
        "(cubic spline) for most data, and ENCLOSING_VOXEL for label data.",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="resampled_%s.nii.gz",
        keep_extension=True,
        argstr="%s",
        position=3,
        desc="the output volume",
    )
    affine = File(
        exists=True,
        mandatory=True,
        argstr="-affine %s",
        position=4,
        desc="the affine file to apply",
    )
    flirt = traits.Bool(
        argstr="-flirt %s %s",
        position=5,
        desc="Set ``True`` if ``affine`` is a FLIRT affine.",
    )
    flirt_source_volume = File(
        exists=True,
        desc="the source volume used when generating the affine; defaults to in_file",
        requires=["flirt"],
    )
    flirt_target_volume = File(
        exists=True,
        desc="the target volume used when generating the affine; defaults to volume_space",
        requires=["flirt"],
    )


class VolumeAffineResampleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output volume")


class VolumeAffineResample(WBCommand):
    """
    Resample a volume file with an affine transformation.

    >>> from nibabies.interfaces.workbench import VolumeAffineResample
    >>> resample = VolumeAffineResample()
    >>> resample.inputs.in_file = data_dir /'functional.nii'
    >>> resample.inputs.volume_space = data_dir /'anatomical.nii'
    >>> resample.inputs.method = 'CUBIC'
    >>> resample.inputs.affine = data_dir / 'func_to_struct.mat'
    >>> resample.cmdline
    'wb_command -volume-resample .../functional.nii .../anatomical.nii CUBIC \
    resampled_functional.nii.gz -affine .../func_to_struct.mat'

    If the affine was generated with FLIRT, this should be indicated.
    By default, the interface will use the ``in_file`` and ``volume_space``
    for references.

    >>> resample.inputs.flirt = True
    >>> resample.cmdline
    'wb_command -volume-resample .../functional.nii .../anatomical.nii CUBIC \
    resampled_functional.nii.gz -affine .../func_to_struct.mat \
    -flirt .../functional.nii .../anatomical.nii'

    However, if other volumes were used to calculate the affine, they can
    be provided:

    >>> resample.inputs.flirt_source_volume = data_dir / 'epi.nii'
    >>> resample.inputs.flirt_target_volume = data_dir /'T1w.nii'
    >>> resample.cmdline
    'wb_command -volume-resample .../functional.nii .../anatomical.nii CUBIC \
    resampled_functional.nii.gz -affine .../func_to_struct.mat \
    -flirt .../epi.nii .../T1w.nii'
    """

    input_spec = VolumeAffineResampleInputSpec
    output_spec = VolumeAffineResampleOutputSpec
    _cmd = "wb_command -volume-resample"

    def _format_arg(self, opt, spec, val):
        if opt == "flirt" and val:
            val = (
                self.inputs.flirt_source_volume or self.inputs.in_file,
                self.inputs.flirt_target_volume or self.inputs.volume_space,
            )
        return super()._format_arg(opt, spec, val)


class VolumeAllLabelsToROIsInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="the input volume label file",
    )
    label_map = traits.Either(
        traits.Int,
        Str,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="the number or name of the label map to use",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="%s_rois.nii.gz",
        argstr="%s",
        position=2,
        desc="the output volume",
    )


class VolumeAllLabelsToROIsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output volume")


class VolumeAllLabelsToROIs(WBCommand):
    """
    Make ROIs from all labels in a volume frame

    The output volume has a frame for each label in the specified input
    frame, other than the ??? label, each of which contains an ROI of all
    voxels that are set to the corresponding label.

    >>> from nibabies.interfaces.workbench import VolumeAllLabelsToROIs
    >>> rois = VolumeAllLabelsToROIs()
    >>> rois.inputs.in_file = data_dir / 'atlas.nii'
    >>> rois.inputs.label_map = 1
    >>> rois.cmdline
    'wb_command -volume-all-labels-to-rois .../atlas.nii 1 atlas_rois.nii.gz'
    """

    input_spec = VolumeAllLabelsToROIsInputSpec
    output_spec = VolumeAllLabelsToROIsOutputSpec
    _cmd = "wb_command -volume-all-labels-to-rois"


class VolumeLabelExportTableInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="the input volume label file",
    )
    label_map = traits.Either(
        traits.Int,
        Str,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="the number or name of the label map to use",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="%s_labels.txt",
        argstr="%s",
        position=2,
        desc="the output text file",
    )


class VolumeLabelExportTableOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output text file")


class VolumeLabelExportTable(WBCommand):
    """
    Export label table from volume as text

    Takes the label table from the volume label map, and writes it to a text
    format matching what is expected by -volume-label-import.

    >>> from nibabies.interfaces.workbench import VolumeLabelExportTable
    >>> label_export = VolumeLabelExportTable()
    >>> label_export.inputs.in_file = data_dir / 'atlas.nii'
    >>> label_export.inputs.label_map = 1
    >>> label_export.cmdline
    'wb_command -volume-label-export-table .../atlas.nii 1 atlas_labels.txt'
    """

    input_spec = VolumeLabelExportTableInputSpec
    output_spec = VolumeLabelExportTableOutputSpec
    _cmd = "wb_command -volume-label-export-table"


class VolumeLabelImportInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="the input volume file",
    )
    label_list_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="text file containing the values and names for labels",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="%s_labels.nii.gz",
        argstr="%s",
        position=2,
        desc="the output workbench label volume",
    )
    discard_others = traits.Bool(
        argstr="-discard-others",
        desc="set any voxels with values not mentioned in the label list to the ??? label",
    )
    unlabeled_values = traits.Int(
        argstr="-unlabeled-value %d",
        desc="the value that will be interpreted as unlabeled",
    )
    subvolume = traits.Either(
        traits.Int,
        Str,
        argstr="-subvolume %s",
        desc="select a single subvolume to import (number or name)",
    )
    drop_unused_labels = traits.Bool(
        argstr="-drop-unused-labels",
        desc="remove any unused label values from the label table",
    )


class VolumeLabelImportOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the output workbench label volume")


class VolumeLabelImport(WBCommand):
    """
    Import a label volume to workbench format

    Creates a label volume from an integer-valued volume file.  The label
    name and color information is stored in the volume header in a nifti
    extension, with a similar format as in caret5, see -volume-help.  You may
    specify the empty string (use "") for <label-list-file>, which will be
    treated as if it is an empty file.  The label list file must have the
    following format (2 lines per label):

    <labelname>
    <key> <red> <green> <blue> <alpha>
    ...

    Label names are specified on a separate line from their value and color,
    in order to let label names contain spaces.  Whitespace is trimmed from
    both ends of the label name, but is kept if it is in the middle of a
    label.  Do not specify the "unlabeled" key in the file, it is assumed
    that 0 means not labeled unless -unlabeled-value is specified.  The value
    of <key> specifies what value in the imported file should be used as this
    label.  The values of <red>, <green>, <blue> and <alpha> must be integers
    from 0 to 255, and will specify the color the label is drawn as (alpha of
    255 means fully opaque, which is probably what you want).

    By default, it will create new label names with names like LABEL_5 for
    any values encountered that are not mentioned in the list file, specify
    -discard-others to instead set these values to the "unlabeled" key.

    >>> from nibabies.interfaces.workbench import VolumeLabelImport
    >>> label_import = VolumeLabelImport()
    >>> label_import.inputs.in_file = data_dir / 'atlas.nii'
    >>> label_import.inputs.label_list_file = data_dir / 'label_list.txt'
    >>> label_import.cmdline
    'wb_command -volume-label-import .../atlas.nii .../label_list.txt \
    atlas_labels.nii.gz'
    """

    input_spec = VolumeLabelImportInputSpec
    output_spec = VolumeLabelImportOutputSpec
    _cmd = "wb_command -volume-label-import"


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
        name_source=["surf_file"],
        name_template="%s_distvol.nii.gz",
        argstr="%s",
        position=2,
        desc="Name of output volume containing signed distances",
    )
    out_mask = File(
        name_source=["surf_file"],
        name_template="%s_distmask.nii.gz",
        argstr="-roi-out %s",
        desc="Name of file to store a mask where the ``out_file`` has a computed value",
    )
    fill_value = traits.Float(
        0.0,
        mandatory=False,
        usedefault=True,
        argstr="-fill-value %f",
        desc="value to put in all voxels that don't get assigned a distance",
    )
    exact_limit = traits.Float(
        5.0,
        usedefault=True,
        argstr="-exact-limit %f",
        desc="distance for exact output in mm",
    )
    approx_limit = traits.Float(
        20.0,
        usedefault=True,
        argstr="-approx-limit %f",
        desc="distance for approximate output in mm",
    )
    approx_neighborhood = traits.Int(
        2,
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
        desc="winding method for point inside surface test",
    )


class CreateSignedDistanceVolumeOutputSpec(TraitedSpec):
    out_file = File(desc="Name of output volume containing signed distances")
    out_mask = File(
        desc="Name of file to store a mask where the ``out_file`` has a computed value"
    )


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
