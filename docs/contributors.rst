.. include:: links.rst

------------------------
Contributing to FMRIPREP
------------------------

This document explains how to prepare a new development environment and
update an existing environment, as necessary.

Development in Docker is encouraged, for the sake of consistency and
portability.
By default, work should be built off of `poldracklab/fmriprep:latest
<https://hub.docker.com/r/poldracklab/fmriprep/>`_ (see the
installation_ guide for the basic procedure for running).

It will be assumed the developer has a working repository in
``$HOME/projects/fmriprep``, and examples are also given for
`niworkflows <https://github.com/poldracklab/niworkflows>`_ and
`nipype`_.

Patching working repositories
=============================
In order to test new code without rebuilding the Docker image, it is
possible to mount working repositories as source directories within the
container.
The `fmriprep-docker`_ script simplifies this for the most common repositories::

    -f PATH, --patch-fmriprep PATH
                          working fmriprep repository (default: None)
    -n PATH, --patch-niworkflows PATH
                          working niworkflows repository (default: None)
    -p PATH, --patch-nipype PATH
                          working nipype repository (default: None)

For instance, if your repositories are contained in ``$HOME/projects``::

    $ fmriprep-docker -f $HOME/projects/fmriprep/fmriprep \
                      -n $HOME/projects/niworkflows/niworkflows \
                      -p $HOME/projects/nipype/nipype \
                      -i poldracklab/fmriprep:latest \
                      $HOME/fullds005 $HOME/dockerout participant

Note the ``-i`` flag allows you to specify an image.

When invoking ``docker`` directly, the mount options must be specified
with the ``-v`` flag::

    -v $HOME/projects/fmriprep/fmriprep:/usr/local/miniconda/lib/python3.6/site-packages/fmriprep:ro
    -v $HOME/projects/niworkflows/niworkflows:/usr/local/miniconda/lib/python3.6/site-packages/niworkflows:ro
    -v $HOME/projects/nipype/nipype:/usr/local/miniconda/lib/python3.6/site-packages/nipype:ro

For example, ::

    $ docker run --rm -v $HOME/fullds005:/data:ro -v $HOME/dockerout:/out \
        -v $HOME/projects/fmriprep/fmriprep:/usr/local/miniconda/lib/python3.6/site-packages/fmriprep:ro \
        poldracklab/fmriprep:latest /data /out/out participant \
        -w /out/work/

In order to work directly in the container, pass the ``--shell`` flag to
``fmriprep-docker``::

    $ fmriprep-docker --shell $HOME/fullds005 $HOME/dockerout participant

This is the equivalent of using ``--entrypoint=bash`` and omitting the fmriprep
arguments in a ``docker`` command::

    $ docker run --rm -v $HOME/fullds005:/data:ro -v $HOME/dockerout:/out \
        -v $HOME/projects/fmriprep/fmriprep:/usr/local/miniconda/lib/python3.6/site-packages/fmriprep:ro --entrypoint=bash \
        poldracklab/fmriprep:latest

Patching containers can be achieved in Singularity by using the PYTHONPATH variable: ::

   $ PYTHONPATH="$HOME/projects/fmriprep" singularity run fmriprep.img \
        /scratch/dataset /scratch/out participant -w /out/work/


Adding dependencies
===================
New dependencies to be inserted into the Docker image will either be
Python or non-Python dependencies.
Python dependencies may be added in three places, depending on whether
the package is large or non-release versions are required.
The image `must be rebuilt <#rebuilding-docker-image>`_ after any
dependency changes.

Python dependencies should generally be included in the ``REQUIRES``
list in `fmriprep/info.py
<https://github.com/poldracklab/fmriprep/blob/29133e5e9f92aae4b23dd897f9733885a60be311/fmriprep/info.py#L46-L61>`_.
If the latest version in `PyPI <https://pypi.org/>`_ is sufficient,
then no further action is required.

For large Python dependencies where there will be a benefit to
pre-compiled binaries, `conda <https://github.com/conda/conda>`_ packages
may also be added to the ``conda install`` line in the `Dockerfile
<https://github.com/poldracklab/fmriprep/blob/29133e5e9f92aae4b23dd897f9733885a60be311/Dockerfile#L46>`_.

Finally, if a specific version of a repository needs to be pinned, edit
the ``requirements.txt`` file.
See the `current
<https://github.com/poldracklab/fmriprep/blob/master/requirements.txt>`_
file for examples.

Non-Python dependencies must also be installed in the Dockerfile, via a
``RUN`` command.
For example, installing an ``apt`` package may be done as follows: ::

    RUN apt-get update && \
        apt-get install -y <PACKAGE>

Rebuilding Docker image
=======================
If it is necessary to rebuild the Docker image, a local image named
``fmriprep`` may be built from within the working fmriprep
repository, located in ``~/projects/fmriprep``: ::

    ~/projects/fmriprep$ docker build -t fmriprep .

To work in this image, replace ``poldracklab/fmriprep:latest`` with
``fmriprep`` in any of the above commands.
This image may be accessed by the `fmriprep-docker`_ wrapper via the
``-i`` flag, e.g. ::

    $ fmriprep-docker -i fmriprep --shell
