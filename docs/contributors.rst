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
`NIPYPE <https://github.com/nipy/nipype>`_.

Patching working repositories
=============================
In order to test new code without rebuilding the Docker image, it is
possible to mount working repositories as source directories within the
container.
In the docker container, the following Python sources are kept in
``/root/src``: ::

    /root/src
    ├── fmriprep/
    ├── nipype/
    └── niworkflows/

To patch in working repositories, for instance contained in
``$HOME/projects/``, add the following arguments to your docker command: ::

    -v $HOME/projects/fmriprep:/root/src/fmriprep:ro
    -v $HOME/projects/niworkflows:/root/src/niworkflows:ro
    -v $HOME/projects/nipype:/root/src/nipype:ro

For example, ::

    $ docker run --rm -v $HOME/fullds005:/data:ro -v $HOME/dockerout:/out \
        -v $HOME/projects/fmriprep:/root/src/fmriprep:ro \
        poldracklab/fmriprep:latest /data /out/out participant \
        -w /out/work/ -t ds005

In order to work directly in the container, use ``--entrypoint=bash``, and
omit the fmriprep arguments: ::

    $ docker run --rm -v $HOME/fullds005:/data:ro -v $HOME/dockerout:/out \
        -v $HOME/projects/fmriprep:/root/src/fmriprep:ro --entrypoint=bash \
        poldracklab/fmriprep:latest

Preparing repository for patching
---------------------------------
In order to patch a working repository into the docker image, its egg-info
must be built.
The first time this is done, the repository should be mounted read/write,
and be installed in editable mode.
For instance, to prepare to patch in fmriprep, niworkflows and nipype,
all located under ``$HOME/projects``, ::

    $ docker run --rm -it --entrypoint=bash \
        -v $HOME/projects/fmriprep:/root/src/fmriprep \
        -v $HOME/projects/niworkflows:/root/src/niworkflows \
        -v $HOME/projects/nipype:/root/src/nipype \
        poldracklab/fmriprep:latest
    root@03e5df018c5e:~# cd ~/src/fmriprep/
    root@03e5df018c5e:~/src/fmriprep# pip install -e .
    root@03e5df018c5e:~# cd ~/src/niworkflows/
    root@03e5df018c5e:~/src/niworkflows# pip install -e .
    root@03e5df018c5e:~# cd ~/src/nipype/
    root@03e5df018c5e:~/src/nipype# pip install -e .

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
