.. include:: links.rst

------------
Installation
------------

There are four ways to use fmriprep: on the free cloud service OpenNeuro.org,
in a `Docker Container`_, in a `Singularity Container`_, or in a `Manually
Prepared Environment`_.
Using OpenNeuro or a local container method is highly recommended.
Once you are ready to run fmriprep, see Usage_ for details.

OpenNeuro
=========

fmriprep is available on the free cloud platform `OpenNeuro.org
<http://openneuro.org>`_.
After uploading your BIDS-compatible dataset to OpenNeuro you will be able to
run fmriprep for free using OpenNeuro servers.
This is the easiest way to run fmriprep, as there is no installation required.

Docker Container
================

In order to run fmriprep in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_.
Once Docker is installed, the recommended way to run fmriprep is to use the
fmriprep-docker_ wrapper, which requires Python and an Internet connection.

To install::

    $ pip install --user --upgrade fmriprep-docker

When run, ``fmriprep-docker`` will generate a Docker command line for you,
print it out for reporting purposes, and then run the command, e.g.::

    $ fmriprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /path/to_output/dir:/out poldracklab/fmriprep:1.0.0 \
        /data /out participant
    ...

You may also invoke ``docker`` directly::

    $ docker run -ti --rm \
        -v filepath/to/data/dir:/data:ro \
        -v filepath/to/output/dir:/out \
        poldracklab/fmriprep:latest \
        /data /out/out \
        participant

For example: ::

    $ docker run -ti --rm \
        -v $HOME/fullds005:/data:ro \
        -v $HOME/dockerout:/out \
        poldracklab/fmriprep:latest \
        /data /out/out \
        participant \
        --ignore fieldmaps

See `External Dependencies`_ for more information (e.g., specific versions) on
what is included in the latest Docker images.

Singularity Container
=====================

For security reasons, many HPCs (e.g., TACC) do not allow Docker containers, but do allow `Singularity <https://github.com/singularityware/singularity>`_ containers.
In this case, start with a machine (e.g., your personal computer) with Docker installed.
Use `docker2singularity <https://github.com/singularityware/docker2singularity>`_ to create a singularity image. You will need an active internet connection and some time. ::

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v D:\host\path\where\to\output\singularity\image:/output \
        singularityware/docker2singularity \
        poldracklab/fmriprep:latest

Transfer the resulting Singularity image to the HPC, for example, using ``scp``. ::

    $ scp poldracklab_fmriprep_latest-*.img user@hcpserver.edu:/path/to/downloads

If the data to be preprocessed is also on the HPC, you are ready to run fmriprep. ::

    $ singularity run path/to/singularity/image.img \
        path/to/data/dir path/to/output/dir \
        participant \
        --participant-label label

For example: ::

    $ singularity run ~/poldracklab_fmriprep_latest-2016-12-04-5b74ad9a4c4d.img \
        /work/04168/asdf/lonestar/ $WORK/lonestar/output \
        participant \
        --participant-label 387 --nthreads 16 -w $WORK/lonestar/work \
        --ants-nthreads 16

.. note::

   Singularity by default `exposes all environment variables from the host inside the container <https://github.com/singularityware/singularity/issues/445>`_.
   Because of this your host libraries (such as nipype) could be accidentally used instead of the ones inside the container - if they are included in PYTHONPATH.
   To avoid such situation we recommend unsetting PYTHONPATH in production use. For example: ::

      $ PYTHONPATH="" singularity run ~/poldracklab_fmriprep_latest-2016-12-04-5b74ad9a4c4d.img \
        /work/04168/asdf/lonestar/ $WORK/lonestar/output \
        participant \
        --participant-label 387 --nthreads 16 -w $WORK/lonestar/work \
        --ants-nthreads 16

Manually Prepared Environment
=============================

.. note::

   This method is not recommended! Make sure you would rather do this than use a `Docker Container`_ or a `Singularity Container`_.

Make sure all of fmriprep's `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.

If you have pip installed, install fmriprep ::

    $ pip install fmriprep

If you have your data on hand, you are ready to run fmriprep: ::

    $ fmriprep data/dir output/dir participant --participant-label label

External Dependencies
=====================

``fmriprep`` is implemented using nipype_, but it requires some other neuroimaging
software tools:

- FSL_ (version 5.0.9)
- ANTs_ (version 2.2.0 - NeuroDocker build)
- AFNI_ (version Debian-16.2.07)
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.0.0)
- FreeSurfer_ (version 6.0.0)
- `ICA-AROMA <https://github.com/rhr-pruim/ICA-AROMA/>`_ (version 0.4.1-beta)
