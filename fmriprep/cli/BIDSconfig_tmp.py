import os,sys
from subprocess import call
from BIDSgenerator import createBIDS

###########################
# config scripts for BIDSgenerator.py 
##########################

# List of possible filenames of your T1 images (i.e. watershed_spgr.nii.gz, T1weighted.nii.gz, T2image.nii.gz, etc.)
T1w_names = ['','','','']

# List of possible filenames of your bold images (i.e. functional.nii.gz, I_anon.nii.gz, anon_func.nii.gz, etc.)
bold_names = ['','','']

# Full path to your project directory
proj_dir = ''

# Full path to a .nii.gz bold image
example_path_bold = ''

# Full path to a .nii.gz T1w image
example_path_T1w = ''

# ID of example path subject (i.e. 3411)
ID = ''
###########################
# DO NOT TOUCH BELOW THIS LINE
###########################

config_args = {'T1w_names':raw_dir,
               'bold_names':subjectlist,
               'proj_dir':runlist,
	       'ID':ID,
	       'example_path_bold':example_path_bold,
	       'example_path_T1w':example_path_T1w}

BIDSgenerator(config_args)
