#!/bin/bash

# Author: nikhil153
# Date: 11 June 2022

# Example command:
# ./run_fmriprep_patch.sh -b ~/scratch/test_data/bids \
#                -w ~/scratch/test_data/tmp \
#                -s 001 \
#                -f ~/scratch/my_repos/fmriprep/fmriprep \
#                -c ~/scratch/my_containers/fmriprep_codecarbon_v2.1.2.sif \
#                -g "CAN" \
#                -t ~/scratch/templateflow

if [ "$#" -lt 12 ]; then
  echo "Incorrect number of arguments: $#. Please provide paths to following:"
  echo "  BIDS dir (-b): input for fmriprep"
  echo "  working dir (-w): directory for fmriprep processing"
  echo "  subject ID (-s): subject subdirectory inside BIDS_DIR"
  echo "  fmriprep code dir (-f): directory for local fmriprep code (carbon-trackers branch)"
  echo "  container (-c): singularity container with carbon tracker packages and dependencies"
  echo "  geolocation (-g): country code used by CodeCarbon to estimate emissions e.g. CAN"
  echo "  templateflow (-t): templateflow dir (optional)"
  echo ""
  echo "Note: the freesurfer license.txt must be inside the working_dir"
  exit 1
fi

while getopts b:w:s:f:c:g:t: flag
do
    case "${flag}" in
        b) BIDS_DIR=${OPTARG};;
        w) WORKING_DIR=${OPTARG};;
        s) SUB_ID=${OPTARG};;
        f) FMRIPREP_CODE=${OPTARG};;
        c) CON_IMG=${OPTARG};;
        g) COUNTRY_CODE=${OPTARG};;
        t) TEMPLATEFLOW_DIR=${OPTARG};;        
    esac
done

echo ""
echo "------------------------------"
echo "Checking arguments provided..."
echo "------------------------------"
echo ""

if [ ! -z $TEMPLATEFLOW_DIR ] 
then 
    echo "Using templates from local templateflow dir: $TEMPLATEFLOW_DIR"
else
    echo "Local templateflow dir not specified. Templates will be downloaded..."
    TEMPLATEFLOW_DIR="Not provided"
fi

echo "
      BIDS dir: $BIDS_DIR 
      working dir: $WORKING_DIR
      subject id: $SUB_ID
      fmriprep code dir: $FMRIPREP_CODE
      container: $CON_IMG
      templateflow: $TEMPLATEFLOW_DIR
      geolocation: $COUNTRY_CODE
      "

DERIVS_DIR=${WORKING_DIR}/output

LOG_FILE=${WORKING_DIR}_fmriprep_anat.log

echo ""
echo "-------------------------------------------------"
echo "Starting fmriprep proc with container: ${CON_IMG}"
echo "-------------------------------------------------"
echo ""

# Create subject specific dirs
FMRIPREP_HOME=${DERIVS_DIR}/fmriprep_home_${SUB_ID}
echo "-------------------------------------------------"
echo "Processing: ${SUB_ID} with home dir: ${FMRIPREP_HOME}"
echo ""
mkdir -p ${FMRIPREP_HOME}

LOCAL_FREESURFER_DIR="${DERIVS_DIR}/freesurfer-6.0.1"
mkdir -p ${LOCAL_FREESURFER_DIR}

# Prepare some writeable bind-mount points.
FMRIPREP_HOST_CACHE=$FMRIPREP_HOME/.cache/fmriprep
mkdir -p ${FMRIPREP_HOST_CACHE}

# Make sure FS_LICENSE is defined in the container.
mkdir -p $FMRIPREP_HOME/.freesurfer
export SINGULARITY_FS_LICENSE=$FMRIPREP_HOME/.freesurfer/license.txt
cp ${WORKING_DIR}/license.txt ${SINGULARITY_FS_LICENSE}

# Designate a templateflow bind-mount point
export SINGULARITY_TEMPLATEFLOW_DIR="/templateflow"

# Singularity CMD 
if [[ $TEMPLATEFLOW_DIR == "Not provided" ]]; then
  SINGULARITY_CMD="singularity run \
  -B ${BIDS_DIR}:/data_dir \
  -B ${FMRIPREP_CODE}:/opt/conda/lib/python3.9/site-packages/fmriprep:ro \
  -B ${FMRIPREP_HOME}:/home/fmriprep --home /home/fmriprep --cleanenv \
  -B ${DERIVS_DIR}:/output \
  -B ${WORKING_DIR}:/work \
  -B ${LOCAL_FREESURFER_DIR}:/fsdir \
  ${CON_IMG}"
else
  echo ""
  echo "Mounting templateflow dir ($TEMPLATEFLOW_DIR) onto container"
  SINGULARITY_CMD="singularity run \
  -B ${BIDS_DIR}:/data_dir \
  -B ${FMRIPREP_CODE}:/opt/conda/lib/python3.9/site-packages/fmriprep:ro \
  -B ${FMRIPREP_HOME}:/home/fmriprep --home /home/fmriprep --cleanenv \
  -B ${DERIVS_DIR}:/output \
  -B ${TEMPLATEFLOW_DIR}:${SINGULARITY_TEMPLATEFLOW_DIR} \
  -B ${WORKING_DIR}:/work \
  -B ${LOCAL_FREESURFER_DIR}:/fsdir \
  ${CON_IMG}"
fi

# Remove IsRunning files from FreeSurfer
find ${LOCAL_FREESURFER_DIR}/sub-$SUB_ID/ -name "*IsRunning*" -type f -delete

# Compose the command line
cmd="${SINGULARITY_CMD} /data_dir /output participant --participant-label $SUB_ID \
-w /work --output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 \
--fs-subjects-dir /fsdir \
--anat-only \
--skip_bids_validation \
--bids-filter-file /data_dir/sample_bids_filter.json \
--fs-license-file /home/fmriprep/.freesurfer/license.txt \
--return-all-components -v \
--write-graph --track-carbon --country-code $COUNTRY_CODE --notrack"

# Optional cmds
#--bids-filter-file ${BIDS_FILTER} --anat-only 

# Setup done, run the command
unset PYTHONPATH
echo ""
echo "--------------------"
echo "Singularity command:"
echo "--------------------"
echo ""
echo Commandline: $cmd
echo ""

eval $cmd
exitcode=$?

exit $exitcode

echo ""
echo "-----------------------"
echo "fmriprep run completed!"
echo "-----------------------"
echo ""