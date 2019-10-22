.. include:: links.rst

.. _run_docker:

Running *fMRIPrep* via Docker containers
========================================
For every new version of *fMRIPrep* that is released, a corresponding Docker
image is generated.
The Docker image *becomes* a container when the execution engine loads the
image and adds an extra layer that makes it *runnable*.
In order to run *fMRIPrep* Docker images, the Docker Engine must be installed
(see :ref:`installation_docker`).

If you used *fMRIPrep* via Docker in the past, you might need to pull down a
more recent version of the image: ::

    $ docker pull poldracklab/fmriprep:<latest-version>

You can run *fMRIPrep* interacting directly with the Docker Engine via the `docker run`
command line, or you can use a lightweight wrapper we created for convenience.

.. _docker_wrapper:

Running *fMRIPrep* with the ``fmriprep-docker`` wrapper
-------------------------------------------------------
Before starting, make sure you
`have the wrapper installed <installation.html#the-fmriprep-docker-wrapper>`__.
When you run ``fmriprep-docker``, it will generate a Docker command line for you,
print it out for reporting purposes, and then execute it without further action
needed, e.g.::

    $ fmriprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /path/to_output/dir:/out poldracklab/fmriprep:1.0.0 \
        /data /out participant
    ...

``fmriprep-docker`` accepts all of the typical options for *fMRIPrep* (see :ref:`usage`),
automatically translating directories into Docker mount points.

We have published a `step-by-step tutorial
<http://reproducibility.stanford.edu/fmriprep-tutorial-running-the-docker-image/>`_
illustrating how to run ``fmriprep-docker``. 
This tutorial also provides valuable troubleshooting insights and advice on 
what to do after *fMRIPrep* has run.


Running *fMRIPrep* directly interacting with the Docker Engine
--------------------------------------------------------------
If you need a finer control over the container execution, or you feel comfortable
with the Docker Engine, avoiding the extra software layer of the wrapper might be
a good decision.

**Accessing filesystems in the host within the container**.
Containers are confined in a sandbox, so they can't access the host in any ways unless 
you explicitly prescribe acceptable accesses to the host.
The Docker Engine provides mounting filesystems into the container with the ``-v`` argument
and the following syntax: ``-v some/path/in/host:/absolute/path/within/container:ro``,
where the trailing ``:ro`` specifies that the mount is read-only.
The mount permissions modifiers can be omitted, which means the mount will have read-write
permissions.
In general, you'll want to at least provide two mount-points: one set in read-only mode
for the input data and one read/write to store the outputs.
Potentially, you'll want to provide one or two more mount-points: one for the working
directory, in case you need to debug some issue or reuse pre-cached results; and
a :ref:`TemplateFlow` folder to preempt the download of your favorite templates in every
run.

**Running containers as a user**.
By default, Docker will run the container as **root**.
Some share systems my limit this feature and only allow running containers as a user.
When the container is run as **root**, files written out to filesystems mounted from the
host will have the user id ``1000`` by default.
In other words, you'll need to be able to run as root in the host to change permissions
or manage these files.
Alternatively, running as a user allows preempting these permissions issues.
It is possible to run as a user with the ``-u`` argument.
In general, we will want to use the same user ID as the running user in the host to
ensure the ownership of files written during the container execution.
Therefore, you will generally run the container with ``-u $( id -u )``.

You may also invoke ``docker`` directly::

    $ docker run -ti --rm \
        -v path/to/data:/data:ro \
        -v path/to/output:/out \
        poldracklab/fmriprep:<latest-version> \
        /data /out/out \
        participant

For example: ::

    $ docker run -ti --rm \
        -v $HOME/ds005:/data:ro \
        -v $HOME/ds005/derivatives:/out \
        -v $HOME/tmp/ds005-workdir:/work \
        poldracklab/fmriprep:<latest-version> \
        /data /out/fmriprep-<latest-version> \
        participant \
        -w /work

Once the Docker Engine arguments are written, the remainder of the command line follows
the :ref:`usage`.
In other words, the first section of the command line is all equivalent to the ``fmriprep``
executable in a bare-metal installation: ::

    $ docker run -ti --rm \                      | These lines
        -v $HOME/ds005:/data:ro \                | are equivalent
        -v $HOME/ds005/derivatives:/out \        | to the fmriprep
        -v $HOME/tmp/ds005-workdir:/work \       | executable.
        poldracklab/fmriprep:<latest-version> \  |
        \
        /data /out/fmriprep-<latest-version> \   | These lines
        participant \                            | correspond to
        -w /work                                 | fmriprep arguments.
