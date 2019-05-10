.. include:: links.rst

------------
Installation
------------

There are four ways to use fmriprep: on the free cloud service OpenNeuro.org,
in a `Docker Container`_, in a `Singularity Container`_, or in a `Manually
Prepared Environment (Python 3.5+)`_.
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
``fmriprep-docker`` is a command that allows you to write your command line 
as if you were running ``fmriprep`` directly, and converts it into a ``docker`` 
command.

To install::

    $ pip install --user --upgrade fmriprep-docker

When you run ``fmriprep-docker``, it will generate a Docker command line for you,
print it out for reporting purposes, and then execute it without further action
needed, e.g.::

    $ fmriprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /path/to_output/dir:/out poldracklab/fmriprep:1.0.0 \
        /data /out participant
    ...

``fmriprep-docker`` accepts all of the typical options for ``fmriprep``,
automatically translating directories into Docker mount points.

We have published a `step-by-step tutorial
<http://reproducibility.stanford.edu/fmriprep-tutorial-running-the-docker-image/>`_
illustrating how to run ``fmriprep-docker``. 
This tutorial also provides valuable troubleshooting insights and advice on 
what to do after fmriprep has run.

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

For security reasons, many HPCs (e.g., TACC) do not allow Docker containers, but do
allow `Singularity <https://github.com/singularityware/singularity>`_ containers.

Preparing a Singularity image (Singularity version >= 2.5)
----------------------------------------------------------
If the version of Singularity on your HPC is modern enough you can create Singularity
image directly on the HCP.
This is as simple as: ::

    $ singularity build /my_images/fmriprep-<version>.simg docker://poldracklab/fmriprep:<version>

Where ``<version>`` should be replaced with the desired version of fMRIPrep that you want to download.


Preparing a Singularity image (Singularity version < 2.5)
---------------------------------------------------------
In this case, start with a machine (e.g., your personal computer) with Docker installed.
Use `docker2singularity <https://github.com/singularityware/docker2singularity>`_ to 
create a singularity image.
You will need an active internet connection and some time. ::

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v D:\host\path\where\to\output\singularity\image:/output \
        singularityware/docker2singularity \
        poldracklab/fmriprep:<version>

Where ``<version>`` should be replaced with the desired version of fMRIPrep that you want 
to download.

Beware of the back slashes, expected for Windows systems.
For \*nix users the command translates as follows: ::

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /absolute/path/to/output/folder:/output \
        singularityware/docker2singularity \
        poldracklab/fmriprep:<version>


Transfer the resulting Singularity image to the HPC, for example, using ``scp``. ::

    $ scp poldracklab_fmriprep*.img user@hcpserver.edu:/my_images

Running a Singularity Image
---------------------------

If the data to be preprocessed is also on the HPC, you are ready to run fmriprep. ::

    $ singularity run --cleanenv /my_images/fmriprep-1.1.2.simg \
        path/to/data/dir path/to/output/dir \
        participant \
        --participant-label label

.. note::

   Singularity by default `exposes all environment variables from the host inside 
   the container <https://github.com/singularityware/singularity/issues/445>`_.
   Because of this your host libraries (such as nipype) could be accidentally used 
   instead of the ones inside the container - if they are included in ``PYTHONPATH``.
   To avoid such situation we recommend using the ``--cleanenv`` singularity flag 
   in production use. For example: ::

      $ singularity run --cleanenv ~/poldracklab_fmriprep_latest-2016-12-04-5b74ad9a4c4d.img \
        /work/04168/asdf/lonestar/ $WORK/lonestar/output \
        participant \
        --participant-label 387 --nthreads 16 -w $WORK/lonestar/work \
        --omp-nthreads 16


   or, unset the ``PYTHONPATH`` variable before running: ::

      $ unset PYTHONPATH; singularity run ~/poldracklab_fmriprep_latest-2016-12-04-5b74ad9a4c4d.img \
        /work/04168/asdf/lonestar/ $WORK/lonestar/output \
        participant \
        --participant-label 387 --nthreads 16 -w $WORK/lonestar/work \
        --omp-nthreads 16


.. note::

   Depending on how Singularity is configured on your cluster it might or might not 
   automatically bind (mount or expose) host folders to the container. 
   If this is not done automatically you will need to bind the necessary folders using 
   the ``-B <host_folder>:<container_folder>`` Singularity argument.
   For example: ::

      $ singularity run --cleanenv -B /work:/work ~/poldracklab_fmriprep_latest-2016-12-04-5b74ad9a4c4d.simg \
        /work/my_dataset/ /work/my_dataset/derivatives/fmriprep \
        participant \
        --participant-label 387 --nthreads 16 \
        --omp-nthreads 16

Manually Prepared Environment (Python 3.5+)
===========================================

.. warning::

   This method is not recommended! Make sure you would rather do this than 
   use a `Docker Container`_ or a `Singularity Container`_.

Make sure all of fmriprep's `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.
A relatively interpretable description of how your environment can be set-up
is found in the `Dockerfile <https://github.com/poldracklab/fmriprep/blob/master/Dockerfile>`_.
As an additional installation setting, FreeSurfer requires a license file (see :ref:`fs_license`).

On a functional Python 3.5 (or above) environment with ``pip`` installed,
fMRIPrep can be installed using the habitual command ::

    $ pip install fmriprep

Check your installation with the ``--version`` argument ::

    $ fmriprep --version


External Dependencies
---------------------

FMRIPrep is written using Python 3.5 (or above), and is based on
nipype_.

FMRIPrep requires some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``fmriprep`` package:

- FSL_ (version 5.0.9)
- ANTs_ (version 2.2.0 - NeuroDocker build)
- AFNI_ (version Debian-16.2.07)
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.0.0)
- FreeSurfer_ (version 6.0.1)
- `ICA-AROMA <https://github.com/rhr-pruim/ICA-AROMA/>`_ (version 0.4.1-beta)
- `bids-validator <https://github.com/bids-standard/bids-validator>`_ (version 1.1.0)


.. _fs_license:

The FreeSurfer license
======================

FMRIPREP uses FreeSurfer tools, which require a license to run.

To obtain a FreeSurfer license, simply register for free at
https://surfer.nmr.mgh.harvard.edu/registration.html.

When using manually-prepared environments or singularity, FreeSurfer will search 
for a license key file first using the ``$FS_LICENSE`` environment variable and then 
in the default path to the license key file (``$FREESURFER_HOME/license.txt``). 
If using the ``--cleanenv`` flag and ``$FS_LICENSE`` is set, use ``--fs-license-file $FS_LICENSE`` 
to pass the license file location to fMRIPrep.

It is possible to run the docker container pointing the image to a local path
where a valid license file is stored.
For example, if the license is stored in the ``$HOME/.licenses/freesurfer/license.txt``
file on the host system: ::

    $ docker run -ti --rm \
        -v $HOME/fullds005:/data:ro \
        -v $HOME/dockerout:/out \
        -v $HOME/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        poldracklab/fmriprep:latest \
        /data /out/out \
        participant \
        --ignore fieldmaps

Using FreeSurfer can also be enabled when using ``fmriprep-docker``: ::

    $ fmriprep-docker --fs-license-file $HOME/.licenses/freesurfer/license.txt \
        /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /home/user/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        -v /path/to_output/dir:/out poldracklab/fmriprep:1.0.0 \
        /data /out participant
    ...

If the environment variable ``$FS_LICENSE`` is set in the host system, then
it will automatically used by ``fmriprep-docker``. For instance, the following
would be equivalent to the latest example: ::

    $ export FS_LICENSE=$HOME/.licenses/freesurfer/license.txt
    $ fmriprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /home/user/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        -v /path/to_output/dir:/out poldracklab/fmriprep:1.0.0 \
        /data /out participant
    ...
