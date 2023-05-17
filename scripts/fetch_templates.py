#!/usr/bin/env python
"""
Standalone script to facilitate caching of required TemplateFlow templates.

To download and view how to use this script, run the following commands inside a terminal:
1. wget https://raw.githubusercontent.com/nipreps/fmriprep/master/scripts/fetch_templates.py
2. python fetch_templates.py -h
"""

import argparse
import os


def fetch_MNI2009():
    """
    Expected templates:

    tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz
    tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz
    tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz
    tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz
    tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-carpet_dseg.nii.gz
    tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-fMRIPrep_boldref.nii.gz
    tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_label-brain_probseg.nii.gz
    """
    template = "MNI152NLin2009cAsym"

    tf.get(template, resolution=(1, 2), desc=None, suffix="T1w")
    tf.get(template, resolution=(1, 2), desc="brain", suffix="mask")
    tf.get(template, resolution=1, atlas=None, desc="carpet", suffix="dseg")
    tf.get(template, resolution=2, desc="fMRIPrep", suffix="boldref")
    tf.get(template, resolution=1, label="brain", suffix="probseg")


def fetch_MNI6():
    """
    Expected templates:

    tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz
    tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz
    tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz
    tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz
    tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz
    """
    template = "MNI152NLin6Asym"

    tf.get(template, resolution=(1, 2), desc=None, suffix="T1w")
    tf.get(template, resolution=(1, 2), desc="brain", suffix="mask")
    # CIFTI
    tf.get(template, resolution=2, atlas="HCP", suffix="dseg")


def fetch_OASIS():
    """
    Expected templates:

    tpl-OASIS30ANTs/tpl-OASIS30ANTs_res-01_T1w.nii.gz
    tpl-OASIS30ANTs/tpl-OASIS30ANTs_res-01_label-WM_probseg.nii.gz
    tpl-OASIS30ANTs/tpl-OASIS30ANTs_res-01_label-BS_probseg.nii.gz
    tpl-OASIS30ANTs/tpl-OASIS30ANTs_res-01_label-brain_probseg.nii.gz
    tpl-OASIS30ANTs/tpl-OASIS30ANTs_res-01_desc-brain_mask.nii.gz
    tpl-OASIS30ANTs/tpl-OASIS30ANTs_res-01_desc-BrainCerebellumExtraction_mask.nii.gz
    """
    template = "OASIS30ANTs"

    tf.get(template, resolution=1, desc=None, label=None, suffix="T1w")
    tf.get(template, resolution=1, label="WM", suffix="probseg")
    tf.get(template, resolution=1, label="BS", suffix="probseg")
    tf.get(template, resolution=1, label="brain", suffix="probseg")
    tf.get(template, resolution=1, label="brain", suffix="mask")
    tf.get(template, resolution=1, desc="BrainCerebellumExtraction", suffix="mask")


def fetch_fsaverage():
    """
    Expected templates:

    tpl-fsaverage/tpl-fsaverage_hemi-L_den-164k_desc-std_sphere.surf.gii
    tpl-fsaverage/tpl-fsaverage_hemi-R_den-164k_desc-std_sphere.surf.gii
    tpl-fsaverage/tpl-fsaverage_hemi-L_den-164k_desc-vaavg_midthickness.shape.gii
    tpl-fsaverage/tpl-fsaverage_hemi-R_den-164k_desc-vaavg_midthickness.shape.gii
    tpl-fsaverage/tpl-fsaverage_hemi-L_den-164k_midthickness.surf.gii
    tpl-fsaverage/tpl-fsaverage_hemi-R_den-164k_midthickness.surf.gii
    """
    template = "fsaverage"

    tf.get(template, density="164k", desc="std", suffix="sphere")
    tf.get(template, density="164k", suffix="midthickness")


def fetch_fsLR():
    """
    Expected templates:

    tpl-fsLR/tpl-fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii
    tpl-fsLR/tpl-fsLR_hemi-L_den-32k_desc-vaavg_midthickness.shape.gii
    tpl-fsLR/tpl-fsLR_hemi-L_den-32k_sphere.surf.gii
    tpl-fsLR/tpl-fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii
    tpl-fsLR/tpl-fsLR_hemi-R_den-32k_desc-vaavg_midthickness.shape.gii
    tpl-fsLR/tpl-fsLR_hemi-R_den-32k_sphere.surf.gii
    tpl-fsLR/tpl-fsLR_space-fsaverage_hemi-L_den-32k_sphere.surf.gii
    tpl-fsLR/tpl-fsLR_space-fsaverage_hemi-R_den-32k_sphere.surf.gii
    """
    tf.get("fsLR", density="32k")


def fetch_all():
    fetch_MNI2009()
    fetch_MNI6()
    fetch_OASIS()
    fetch_fsaverage()
    fetch_fsLR()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helper script for pre-caching required templates to run fMRIPrep",
    )
    parser.add_argument(
        "--tf-dir",
        type=os.path.abspath,
        help="Directory to save templates in. If not provided, templates will be saved to"
        " `${HOME}/.cache/templateflow`.",
    )
    opts = parser.parse_args()

    # set envvar (if necessary) prior to templateflow import
    if opts.tf_dir is not None:
        os.environ["TEMPLATEFLOW_HOME"] = opts.tf_dir

    import templateflow.api as tf

    fetch_all()
