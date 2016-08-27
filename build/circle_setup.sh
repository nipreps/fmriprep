#!/bin/bash

set -x
set -e

# Get test data
if [[ ! -d ${HOME}/scratch/data/aa_conn ]]; then
    # Folder for downloads
    mkdir -p ${HOME}/downloads
    wget -c -O ${HOME}/downloads/AA_Conn.tar "http://googledrive.com/host/0BxI12kyv2olZY0ZkZVB4TUVtSWM"
    mkdir -p ${HOME}/scratch/data/
    tar xf ${HOME}/downloads/AA_Conn.tar -C ${HOME}/scratch/data
fi

echo "{plugin: MultiProc, plugin_args: {n_proc: 4}}" > ${HOME}/scratch/plugin.yml
