#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel

# Compile wheels
for PYBIN in /opt/python/cp3{6,7}*/bin; do
    "${PYBIN}/pip" install -U pip
    "${PYBIN}/pip" wheel /io/ -w dist/
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/dist/
done

# Install packages
################## Disabled bc pybids gives no distribution found
# for PYBIN in /opt/python/cp3{6,7}*/bin; do
#     "${PYBIN}/pip" install "$PKGNAME" --no-index -f /io/dist
# done
