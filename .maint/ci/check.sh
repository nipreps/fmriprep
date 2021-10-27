#!/bin/bash

echo Running tests

source .maint/ci/activate.sh

set -eu

# Required variables
echo CHECK_TYPE = $CHECK_TYPE

set -x

if [ "${CHECK_TYPE}" == "doc" ]; then
    cd doc
    make html && make doctest
elif [ "${CHECK_TYPE}" == "tests" ]; then
    pytest --doctest-modules --cov fmriprep --cov-report xml \
        --junitxml=test-results.xml -v fmriprep
else
    false
fi

set +eux

echo Done running tests
