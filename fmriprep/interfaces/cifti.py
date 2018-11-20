#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling connectivity
~~~~~~~~~~~~~~~~~~~~~
Combines FreeSurfer surfaces with subcortical volumes

"""
import os
from glob import glob
import json

import nibabel as nb
from nibabel import cifti2 as ci
import numpy as np
from nilearn.image import resample_to_img

from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, traits,
    SimpleInterface, Directory
)
from niworkflows.data import getters
from nipype.utils.filemanip import split_filename

# CITFI structures with corresponding FS labels
CIFTI_STRUCT_WITH_LABELS = {
    # SURFACES
    'CIFTI_STRUCTURE_CORTEX_LEFT': None,
    'CIFTI_STRUCTURE_CORTEX_RIGHT': None,

    # SUBCORTICAL
    'CIFTI_STRUCTURE_ACCUMBENS_LEFT': [26],
    'CIFTI_STRUCTURE_ACCUMBENS_RIGHT': [58],
    'CIFTI_STRUCTURE_AMYGDALA_LEFT': [18],
    'CIFTI_STRUCTURE_AMYGDALA_RIGHT': [54],
    'CIFTI_STRUCTURE_BRAIN_STEM': [16],
    'CIFTI_STRUCTURE_CAUDATE_LEFT': [11],
    'CIFTI_STRUCTURE_CAUDATE_RIGHT': [50],
    'CIFTI_STRUCTURE_CEREBELLUM_LEFT': [6],
    'CIFTI_STRUCTURE_CEREBELLUM_RIGHT': [45],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT': [28],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT': [60],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT': [17],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT': [53],
    'CIFTI_STRUCTURE_PALLIDUM_LEFT': [13],
    'CIFTI_STRUCTURE_PALLIDUM_RIGHT': [52],
    'CIFTI_STRUCTURE_PUTAMEN_LEFT': [12],
    'CIFTI_STRUCTURE_PUTAMEN_RIGHT': [51],
    'CIFTI_STRUCTURE_THALAMUS_LEFT': [10],
    'CIFTI_STRUCTURE_THALAMUS_RIGHT': [49],
}


class GenerateCiftiInputSpec(BaseInterfaceInputSpec):
    bold_file = File(mandatory=True, exists=True, desc="input BOLD file")
    volume_target = traits.Enum("MNI152NLin2009cAsym", mandatory=True, usedefault=True,
                                desc="CIFTI volumetric output space")
    surface_target = traits.Enum("fsaverage5", "fsaverage6", mandatory=True,
                                 usedefault=True, desc="CIFTI surface target space")
    subjects_dir = Directory(mandatory=True, desc="FreeSurfer SUBJECTS_DIR")
    TR = traits.Float(mandatory=True, desc="repetition time")
    gifti_files = traits.List(File(exists=True), mandatory=True,
                              desc="list of surface geometry files (length 2 with order [L,R])")


class GenerateCiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="generated CIFTI file")
    variant = traits.Str(desc="combination of target spaces label")
    variant_key = File(exists=True, desc='file storing variant space information')


class GenerateCifti(SimpleInterface):
    """
    Generate CIFTI image from BOLD file in target spaces. Currently supported

    * target surfaces: fsaverage5, fsaverage6
    * target volumes: OASIS-TRT-20_DKT31 labels in MNI152NLin2009cAsym
    """
    input_spec = GenerateCiftiInputSpec
    output_spec = GenerateCiftiOutputSpec

    def _run_interface(self, runtime):
        self._results["variant_key"], self._results["variant"] = self._define_variant()
        annotation_files, label_file, download_link = self._fetch_data()
        self._results["out_file"] = self._create_cifti_image(
            self.inputs.bold_file,
            label_file,
            annotation_files,
            self.inputs.gifti_files,
            self.inputs.volume_target,
            self.inputs.surface_target,
            self.inputs.TR,
            download_link)
        return runtime

    def _define_variant(self):
        """Assign arbitrary label to combination of CIFTI spaces"""
        space = None
        variants = {
            # to be expanded once addtional spaces are supported
            'space1': ['fsaverage5', 'MNI152NLin2009cAsym'],
            'space2': ['fsaverage6', 'MNI152NLin2009cAsym'],
        }
        for sp, targets in variants.items():
            if all(target in targets for target in
                    [self.inputs.surface_target, self.inputs.volume_target]):
                space = sp
        if space is None:
            raise NotImplementedError

        variant_key = os.path.abspath('dtseries_variant.json')
        with open(variant_key, 'w') as fp:
            json.dump({space: variants[space]}, fp)
        return variant_key, space

    def _fetch_data(self):
        """Converts inputspec to files"""
        if (self.inputs.surface_target == "fsnative" or
                self.inputs.volume_target != "MNI152NLin2009cAsym"):
            # subject space is not support yet
            raise NotImplementedError

        annotation_files = sorted(glob(os.path.join(self.inputs.subjects_dir,
                                                    self.inputs.surface_target,
                                                    'label',
                                                    '*h.aparc.annot')))
        if not annotation_files:
            raise IOError("Freesurfer annotations for %s not found in %s" % (
                          self.inputs.surface_target, self.inputs.subjects_dir))

        label_space = 'OASISTRT20'
        label_file = str(getters.get_template(label_space) /
                         'tpl-OASISTRT20_variant-DKT31_space-MNI152NLin2009cAsym.nii.gz')

        download_link = getters.OSF_PROJECT_URL + getters.OSF_RESOURCES[label_space][0]
        return annotation_files, label_file, download_link

    @staticmethod
    def _create_cifti_image(bold_file, label_file, annotation_files, gii_files,
                            volume_target, surface_target, tr, download_link=None):
        """
        Generate CIFTI image in target space

        Parameters
            bold_file : 4D BOLD timeseries
            label_file : label atlas
            annotation_files : FreeSurfer annotations
            gii_files : 4D BOLD surface timeseries in GIFTI format
            volume_target : label atlas space
            surface_target : gii_files space
            tr : repetition timeseries
            download_link : URL to download label_file

        Returns
            out_file : BOLD data as CIFTI dtseries
        """

        label_img = nb.load(label_file)
        bold_img = resample_to_img(bold_file, label_img)

        bold_data = bold_img.get_data()
        timepoints = bold_img.shape[3]
        label_data = label_img.get_data()

        # set up CIFTI information
        series_map = ci.Cifti2MatrixIndicesMap((0, ),
                                               'CIFTI_INDEX_TYPE_SERIES',
                                               number_of_series_points=timepoints,
                                               series_exponent=0,
                                               series_start=0.0,
                                               series_step=tr,
                                               series_unit='SECOND')
        # Create CIFTI brain models
        idx_offset = 0
        brainmodels = []
        bm_ts = np.empty((timepoints, 0))

        for structure, labels in CIFTI_STRUCT_WITH_LABELS.items():
            if labels is None:  # surface model
                model_type = "CIFTI_MODEL_TYPE_SURFACE"
                # use the corresponding annotation
                hemi = structure.split('_')[-1]
                annot = nb.freesurfer.read_annot(annotation_files[hemi == "RIGHT"])
                # currently only supports L/R cortex
                gii = nb.load(gii_files[hemi == "RIGHT"])
                # calculate total number of vertices
                surf_verts = len(annot[0])
                # remove medial wall for CIFTI format
                vert_idx = np.nonzero(annot[0] != annot[2].index(b'unknown'))[0]
                # extract values across volumes
                ts = np.array([tsarr.data[vert_idx] for tsarr in gii.darrays])

                vert_idx = ci.Cifti2VertexIndices(vert_idx)
                bm = ci.Cifti2BrainModel(index_offset=idx_offset,
                                         index_count=len(vert_idx),
                                         model_type=model_type,
                                         brain_structure=structure,
                                         vertex_indices=vert_idx,
                                         n_surface_vertices=surf_verts)
                bm_ts = np.column_stack((bm_ts, ts))
                idx_offset += len(vert_idx)
                brainmodels.append(bm)
            else:
                model_type = "CIFTI_MODEL_TYPE_VOXELS"
                vox = []
                ts = None
                for label in labels:
                    ijk = np.nonzero(label_data == label)
                    ts = (bold_data[ijk] if ts is None
                          else np.concatenate((ts, bold_data[ijk])))
                    vox += [[ijk[0][ix], ijk[1][ix], ijk[2][ix]]
                            for ix, row in enumerate(ts)]

                bm_ts = np.column_stack((bm_ts, ts.T))

                vox = ci.Cifti2VoxelIndicesIJK(vox)
                bm = ci.Cifti2BrainModel(index_offset=idx_offset,
                                         index_count=len(vox),
                                         model_type=model_type,
                                         brain_structure=structure,
                                         voxel_indices_ijk=vox)
                idx_offset += len(vox)
                brainmodels.append(bm)

        volume = ci.Cifti2Volume(
            bold_img.shape[:3],
            ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, bold_img.affine))
        brainmodels.append(volume)

        # create CIFTI geometry based on brainmodels
        geometry_map = ci.Cifti2MatrixIndicesMap((1, ),
                                                 'CIFTI_INDEX_TYPE_BRAIN_MODELS',
                                                 maps=brainmodels)
        # provide some metadata to CIFTI matrix
        meta = {
            "target_surface": surface_target,
            "target_volume": volume_target,
            "download_link": download_link,
        }
        # generate and save CIFTI image
        matrix = ci.Cifti2Matrix()
        matrix.append(series_map)
        matrix.append(geometry_map)
        matrix.metadata = ci.Cifti2MetaData(meta)
        hdr = ci.Cifti2Header(matrix)
        img = ci.Cifti2Image(bm_ts, hdr)
        img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES')

        _, out_base, _ = split_filename(bold_file)
        out_file = "{}.dtseries.nii".format(out_base)
        ci.save(img, out_file)
        return os.path.join(os.getcwd(), out_file)


class CiftiNameSourceInputSpec(BaseInterfaceInputSpec):
    variant = traits.Str(mandatory=True,
                         desc=('unique label of spaces used in combination to'
                               ' generate CIFTI file'))


class CiftiNameSourceOutputSpec(TraitedSpec):
    out_name = traits.Str(desc='(partial) filename formatted according to template')


class CiftiNameSource(SimpleInterface):
    """
    Construct new filename based on unique label of spaces used to generate a
    CIFTI file
    """
    input_spec = CiftiNameSourceInputSpec
    output_spec = CiftiNameSourceOutputSpec

    def _run_interface(self, runtime):
        suffix = 'bold.dtseries'
        if 'hcp' in self.inputs.variant:
            suffix = 'space-hcp_bold.dtseries'
        self._results['out_name'] = suffix
        return runtime
