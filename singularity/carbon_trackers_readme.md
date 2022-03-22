# Stand-in readme for carbon trackers usage with fmriprep. (This should be merged with official docs after PR). 

## Goal: track power consumption (i.e. cpu power draws) of fmriprep workflow using [CodeCarbon](https://mlco2.github.io/codecarbon/index.html)

## fmriprep branch: carbon-trackers

## files modified:
1. fmriprep/cli/parser.py
2. fmriprep/cli/run.py
3. fmriprep/config.py

## files added:
1. scripts/run_fmriprep_patch.sh
2. singularity/fmriprep_with_carbon_trackers.def
3. singularity/requirements.txt

## env setup
We will be running this test code with a Singularity container. For that we need to 1) include CodeCarbon dependancies and 2) patch this branch onto the container. 

1. Build singularity image from [fmriprep_with_carbon_trackers.def](./fmriprep_with_carbon_trackers.def) to include [CodeCarbon](https://mlco2.github.io/codecarbon/index.html) dependencies. 
2. Clone this [repo](https://github.com/nikhil153/fmriprep/tree/carbon-trackers) and checkout "carbon-trackers" branch
3. Use [run_fmriprep_patch.sh](../scripts/run_fmriprep_patch.sh) to setup paths for
    - singularity image, this "carbon-trackers" branch (patching)
    - bids_dir, output_dir, subject_id etc 
4. Run your workflow using sudo

```
example cmd: sudo ./run_fmriprep_patch.sh /path/to/bids_dir /path/ouput_dir <subject_id> 
```
