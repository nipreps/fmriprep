SETUP_REQUIRES="pip build"

# Numpy and scipy upload nightly/weekly/intermittent wheels
NIGHTLY_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
STAGING_WHEELS="https://pypi.anaconda.org/multibuild-wheels-staging/simple"
PRE_PIP_FLAGS="--pre --extra-index-url $NIGHTLY_WHEELS --extra-index-url $STAGING_WHEELS --prefer-binary"
