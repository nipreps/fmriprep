#!/bin/bash

echo "Building archive"

source .maint/ci/activate.sh

set -eu

# Required dependencies
echo "INSTALL_TYPE = $INSTALL_TYPE"

set -x

if [ "$INSTALL_TYPE" == "sdist" ]; then
    python setup.py egg_info  # check egg_info while we're here
    python setup.py sdist
    export ARCHIVE=$( ls dist/*.tar.gz )
elif [ "$INSTALL_TYPE" == "wheel" ]; then
    python setup.py bdist_wheel
    export ARCHIVE=$( ls dist/*.whl )
elif [ "$INSTALL_TYPE" == "archive" ]; then
    export ARCHIVE="package.tar.gz"
    git archive -o $ARCHIVE HEAD
elif [ "$INSTALL_TYPE" == "pip" ]; then
    export ARCHIVE="."
fi

if [ "${ARCHIVE:0:5}" = "dist/" ]; then
    python -m pip install twine
    python -m twine check $ARCHIVE
fi

set +eux
