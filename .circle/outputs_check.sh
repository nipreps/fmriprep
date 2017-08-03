#!/bin/bash
#
# Checking out fmriprep outputs
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.

# Exit if docs_only tag is found
if echo ${GIT_COMMIT_DESC} | grep -Pi 'docs[ _]?only'; then
    echo "Building [docs_only], nothing to do."
    exit 0
fi

DATASET=ds054

if [ "${CIRCLE_NODE_INDEX:-0}" == "1" ]; then
	DATASET=ds005
fi

echo "Checking outputs (${DATASET})..."
mkdir -p ${HOME}/${DATASET}/test
find $HOME/${DATASET}/out  | sed s+$HOME/++ | sort > ${HOME}/${DATASET}/test/outputs.out
diff $HOME/$CIRCLE_PROJECT_REPONAME/.circle/data/${DATASET}_outputs.txt ${HOME}/${DATASET}/test/outputs.out

exit $?
