.. include:: links.rst

.. _Usage :

Usage Notes
===========
.. warning::
   As of *fMRIPRep* 1.0.12, the software includes a tracking system
   to report usage statistics and errors. Users can opt-out using
   the ``--notrack`` command line argument.


Execution and the BIDS format
-----------------------------
The *fMRIPRep* workflow takes as principal input the path of the dataset
that is to be processed.
The input dataset is required to be in valid :abbr:`BIDS (Brain Imaging Data
Structure)` format, and it must include at least one T1w structural image and
(unless disabled with a flag) a BOLD series.
We highly recommend that you validate your dataset with the free, online
`BIDS Validator <http://bids-standard.github.io/bids-validator/>`_.

The exact command to run *fMRIPRep* depends on the Installation_ method.
The common parts of the command follow the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.
Example: ::

    fmriprep data/bids_root/ out/ participant -w work/


Command-Line Arguments
----------------------
.. argparse::
   :ref: fmriprep.cli.parser._build_parser
   :prog: fmriprep
   :nodefault:
   :nodefaultconst:


The command-line interface of the docker wrapper
------------------------------------------------

.. argparse::
   :ref: fmriprep_docker.get_parser
   :prog: fmriprep-docker
   :nodefault:
   :nodefaultconst:


.. _fs_license:

The FreeSurfer license
----------------------

*fMRIPRep* uses FreeSurfer tools, which require a license to run.

To obtain a FreeSurfer license, simply register for free at
https://surfer.nmr.mgh.harvard.edu/registration.html.

When using manually-prepared environments or singularity, FreeSurfer will search
for a license key file first using the ``$FS_LICENSE`` environment variable and then
in the default path to the license key file (``$FREESURFER_HOME/license.txt``).
If using the ``--cleanenv`` flag and ``$FS_LICENSE`` is set, use ``--fs-license-file $FS_LICENSE``
to pass the license file location to *fMRIPRep*.

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


Troubleshooting
---------------

Logs and crashfiles are outputted into the
``<output dir>/fmriprep/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: http://fmriprep.readthedocs.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/poldracklab/fmriprep/issues.

If you have a problem or would like to ask a question about how to use *fMRIPRep*,
please submit a question to `NeuroStars.org <http://neurostars.org/tags/fmriprep>`_ with an ``fmriprep`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

Previous questions about *fMRIPRep* are available here:
http://neurostars.org/tags/fmriprep/

To participate in the *fMRIPRep* development-related discussions please use the
following mailing list: http://mail.python.org/mailman/listinfo/neuroimaging
Please add *[fmriprep]* to the subject line when posting on the mailing list.
