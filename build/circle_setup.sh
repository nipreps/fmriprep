#!/bin/bash

set -x
set -e

# Get test data
if [[ ! -d ${HOME}/scratch/data/aa_conn ]]; then
    # Folder for downloads
    mkdir -p ${HOME}/downloads
    wget -c -O ${HOME}/downloads/testdata.tar "https://3552243d5be815c1b09152da6525cb8fe7b900a6.googledrive.com/host/0BxI12kyv2olZVUswazA3NkFvOXM/ds054_downsampled.tar.gz"
    mkdir -p ${HOME}/scratch/data/
    tar xf ${HOME}/downloads/testdata.tar -C ${HOME}/scratch/data
fi

echo "{plugin: MultiProc, plugin_args: {n_proc: 4}}" > ${HOME}/scratch/plugin.yml
