.. include:: links.rst

------------------------
Contributing to FMRIPREP
------------------------

This document explains how to prepare a new development environment and
update an existing environment, as necessary.

Development in Docker is encouraged, for the sake of consistency and
portability.
By default, work should be built off of `poldracklab/fmriprep:unstable
<https://hub.docker.com/r/poldracklab/fmriprep/>`_, which tracks the ``master`` branch,
or ``poldracklab/fmriprep:latest``, which tracks the latest release version (see the
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
The `Docker wrapper`_ script simplifies this
for the most common repositories::

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

    -v $HOME/projects/fmriprep/fmriprep:/usr/local/miniconda/lib/python3.7/site-packages/fmriprep:ro
    -v $HOME/projects/niworkflows/niworkflows:/usr/local/miniconda/lib/python3.7/site-packages/niworkflows:ro
    -v $HOME/projects/nipype/nipype:/usr/local/miniconda/lib/python3.7/site-packages/nipype:ro

For example, ::

    $ docker run --rm -v $HOME/fullds005:/data:ro -v $HOME/dockerout:/out \
        -v $HOME/projects/fmriprep/fmriprep:/usr/local/miniconda/lib/python3.7/site-packages/fmriprep:ro \
        poldracklab/fmriprep:latest /data /out/out participant \
        -w /out/work/

In order to work directly in the container, pass the ``--shell`` flag to
``fmriprep-docker``::

    $ fmriprep-docker --shell $HOME/fullds005 $HOME/dockerout participant

This is the equivalent of using ``--entrypoint=bash`` and omitting the fmriprep
arguments in a ``docker`` command::

    $ docker run --rm -v $HOME/fullds005:/data:ro -v $HOME/dockerout:/out \
        -v $HOME/projects/fmriprep/fmriprep:/usr/local/miniconda/lib/python3.7/site-packages/fmriprep:ro --entrypoint=bash \
        poldracklab/fmriprep:latest

Patching containers can be achieved in Singularity analogous to ``docker``
using the ``--bind`` (``-B``) option: ::

    $ singularity run \
        -B $HOME/projects/fmriprep/fmriprep:/usr/local/miniconda/lib/python3.7/site-packages/fmriprep \
        fmriprep.img \
        /scratch/dataset /scratch/out participant -w /out/work/

Or you can patch Singularity containers using the PYTHONPATH variable: ::

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
list in `fmriprep/__about__.py
<https://github.com/poldracklab/fmriprep/blob/510f28db4aab8a6adde0ccadeba2da7d78ed696e/fmriprep/__about__.py#L87-L107>`_.
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

    ~/projects/fmriprep$ VERSION=$( python get_version.py )
    ~/projects/fmriprep$ docker build -t fmriprep --build-arg VERSION=$VERSION .

The ``VERSION`` build argument is necessary to ensure that help text
can be reliably generated. The ``get_version.py`` tool constructs the
version string from the current repository state.

To work in this image, replace ``poldracklab/fmriprep:latest`` with
``fmriprep`` in any of the above commands.
This image may be accessed by the `Docker wrapper`_
via the ``-i`` flag, e.g., ::

    $ fmriprep-docker -i fmriprep --shell

Code-Server Development Environment (Experimental)
==================================================
To get the best of working with containers and having an interactive
development environment, we have an experimental setup with `code-server
<https://github.com/cdr/code-server>`_.

.. Note::
    We have `a video walking through the process
    <https://youtu.be/bkZ-NyUaTvg>`_ if you want a visual guide.

1. Build the Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~
We will use the ``Dockerfile_devel`` file to build 
our development docker image::

    $ cd $HOME/projects/fmriprep
    $ docker build -t fmriprep_devel -f Dockerfile_devel .

2. Run the Docker image
~~~~~~~~~~~~~~~~~~~~~~~
We can start a docker container using the image we built (``fmriprep_devel``)::

    $ docker run -it -p 127.0.0.1:8445:8080 -v ${PWD}:/src/fmriprep fmriprep_devel:latest

.. Note::
    If you are using windows shell, ${PWD} may not be defined, instead use the absolute
    path to your fmriprep directory.

.. Note::
    If you are using Docker-Toolbox, you will need to change your virtualbox settings
    using `these steps as a guide
    <https://github.com/jdkent/tutDockerRstudio#additional-setup-for-docker-toolbox>`_.
    (For step ``6``, instead of ``Name = rstudio; Host Port = 8787; Guest Port = 8787``,
    have ``Name = code-server; Host Port = 8443; Guest Port = 8080``.)
    Then in the docker command above, change ``127.0.0.1:8445:8080``
    to ``192.168.99.100:8445:8080``.

If the container started correctly, you should see the following on your console::

    INFO  Server listening on http://localhost:8080
    INFO    - No authentication
    INFO    - Not serving HTTPS

Now you can switch to your favorite browser and go to: ``127.0.0.1:8445``
(or ``192.168.99.100:8445`` for Docker Toolbox).

3. Copy fmriprep.egg-info into your fmriprep directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``fmriprep.egg-info`` makes the fmriprep package exacutable inside the docker container.
Open a terminal in vscode and type the following::

    $ cp -R /src/fmriprep.egg-info /src/fmriprep/


Code-Server Development Environment Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The editor is `vscode <https://code.visualstudio.com/docs>`_

- There are several preconfigured debugging tests under
  the debugging icon in the activity bar

  - see `vscode debugging python <https://code.visualstudio.com/docs/python/debugging>`_
    for details.

- The ``gitlens`` and ``python`` extensions are preinstalled to improve
  the development experience in vscode.


Adding new features to the citation boilerplate
===============================================

The citation boilerplate is built by adding two dunder attributes
of workflow objects: ``__desc__`` and ``__postdesc__``.
Once the full *fMRIPrep* workflow is built, starting from the
outer workflow and visiting all sub-workflows in topological
order, all defined ``__desc__`` are appended to the citation
boilerplate before descending into sub-workflows.
Once all the sub-workflows of a given workflow have
been visited, then the ``__postdesc__`` attribute is appended
and the execution pops out to higher level workflows.
The dunder attributes are written in Markdown language, and may contain
references.
To add a reference, just add a new Bibtex entry to the references
database (``/fmriprep/data/boilerplate.bib``).
You can then use the Bibtex handle within the Markdown text.
For example, if the Bibtex handle is ``myreference``, a citation
will be generated in Markdown language with ``@myreference``.
To generate citations with parenthesis and/or additional content,
brackets should be used: e.g., ``[see @myreference]`` will produce
a citation like *(see Doe J. et al 2018)*.


An example of how this works is shown here: ::

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
    Head-motion parameters with respect to the BOLD reference
    (transformation matrices, and six corresponding rotation and translation
    parameters) are estimated before any spatiotemporal filtering using
    `mcflirt` [FSL {fsl_ver}, @mcflirt].
    """.format(fsl_ver=fsl.Info().version() or '<ver>')
