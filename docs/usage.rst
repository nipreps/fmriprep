.. include:: links.rst

.. _Usage :

Usage Notes
===========
.. warning::
   As of *fMRIPrep* 1.0.12, the software includes a tracking system
   to report usage statistics and errors. Users can opt-out using
   the ``--notrack`` command line argument.


Execution and the BIDS format
-----------------------------
The *fMRIPrep* workflow takes as principal input the path of the dataset
that is to be processed.
The input dataset is required to be in valid :abbr:`BIDS (Brain Imaging Data
Structure)` format, and it must include at least one T1w structural image and
(unless disabled with a flag) a BOLD series.
We highly recommend that you validate your dataset with the free, online
`BIDS Validator <https://bids-standard.github.io/bids-validator/>`_.

The exact command to run *fMRIPRep* depends on the Installation_ method.
The common parts of the command follow the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.
Example: ::

    fmriprep data/bids_root/ out/ participant -w work/

Further information about BIDS and BIDS-Apps can be found at the
`NiPreps portal <https://www.nipreps.org/apps/framework/>`__.

Command-Line Arguments
----------------------
.. argparse::
   :ref: fmriprep.cli.parser._build_parser
   :prog: fmriprep


The command-line interface of the docker wrapper
------------------------------------------------

.. argparse::
   :ref: fmriprep_docker.__main__.get_parser
   :prog: fmriprep-docker
   :nodefault:
   :nodefaultconst:



Limitations and reasons not to use *fMRIPrep*
---------------------------------------------

1. Very narrow :abbr:`FoV (field-of-view)` images oftentimes do not contain
   enough information for standard image registration methods to work correctly.
   Also, problems may arise when extracting the brain from these data.
   fMRIPrep supports pre-aligned BOLD series, and accepting pre-computed
   derivatives such as brain masks is a target of future effort.
2. *fMRIPrep* may also underperform for particular populations (e.g., infants) and
   non-human brains, although appropriate templates can be provided to overcome the
   issue.
3. The "EPInorm" approach is currently not supported, although we plan to implement
   this feature (see `#620 <https://github.com/nipreps/fmriprep/issues/620>`__).
4. If you really want unlimited flexibility (which is obviously a double-edged sword).
5. If you want students to suffer through implementing each step for didactic purposes,
   or to learn shell-scripting or Python along the way.
6. If you are trying to reproduce some *in-house* lab pipeline.

(Reasons 4-6 were kindly provided by S. Nastase in his
`open review <https://pubpeer.com/publications/6B3E024EAEBF2C80085FDF644C2085>`__
of our `pre-print <https://doi.org/10.1101/306951>`__).

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
        nipreps/fmriprep:latest \
        /data /out/out \
        participant \
        --ignore fieldmaps

Using FreeSurfer can also be enabled when using ``fmriprep-docker``: ::

    $ fmriprep-docker --fs-license-file $HOME/.licenses/freesurfer/license.txt \
        /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /home/user/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        -v /path/to_output/dir:/out nipreps/fmriprep:1.0.0 \
        /data /out participant
    ...

If the environment variable ``$FS_LICENSE`` is set in the host system, then
it will automatically used by ``fmriprep-docker``. For instance, the following
would be equivalent to the latest example: ::

    $ export FS_LICENSE=$HOME/.licenses/freesurfer/license.txt
    $ fmriprep-docker /path/to/data/dir /path/to/output/dir participant
    RUNNING: docker run --rm -it -v /path/to/data/dir:/data:ro \
        -v /home/user/.licenses/freesurfer/license.txt:/opt/freesurfer/license.txt \
        -v /path/to_output/dir:/out nipreps/fmriprep:1.0.0 \
        /data /out participant
    ...


.. _prev_derivs:

Reusing precomputed derivatives
-------------------------------

Reusing a previous, partial execution of *fMRIPrep*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*fMRIPrep* will pick up where it left off a previous execution, so long as the work directory
points to the same location, and this directory has not been changed/manipulated.
Some workflow nodes will rerun unconditionally, so there will always be some amount of
reprocessing.

Using a previous run of *FreeSurfer*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*fMRIPrep* will automatically reuse previous runs of *FreeSurfer* if a subject directory
named ``freesurfer/`` is found in the output directory (``<output_dir>/freesurfer``).
Reconstructions for each participant will be checked for completeness, and any missing
components will be recomputed.
You can use the ``--fs-subjects-dir`` flag to specify a different location to save
FreeSurfer outputs.
If precomputed results are found, they will be reused.

BIDS Derivatives reuse
~~~~~~~~~~~~~~~~~~~~~~
As of version 23.2.0, *fMRIPrep* can reuse precomputed derivatives that follow BIDS Derivatives
conventions. To provide derivatives to *fMRIPrep*, use the ``--derivatives`` (``-d``) flag one
or more times.

This mechanism replaces the earlier, more limited ``--anat-derivatives`` flag.

.. note::
   Derivatives reuse is considered *experimental*.

This feature has several intended use-cases:

  * To enable fMRIPrep to be run in a "minimal" mode, where only the most essential
    derivatives are generated. This can be useful for large datasets where disk space
    is a concern, or for users who only need a subset of the derivatives. Further
    derivatives may be generated later, or by a different tool.
  * To enable fMRIPrep to be integrated into a larger processing pipeline, where
    other tools may generate derivatives that fMRIPrep can use in place of its own
    steps.
  * To enable users to substitute their own custom derivatives for those generated
    by fMRIPrep. For example, a user may wish to use a different brain extraction
    tool, or a different registration tool, and then use fMRIPrep to generate the
    remaining derivatives.
  * To enable complicated meta-workflows, where fMRIPrep is run multiple times
    with different options, and the results are combined. For example, the
    `My Connectome <https://openneuro.org/datasets/ds000031/>`__ dataset contains
    107 sessions for a single-subject. Processing of all sessions simultaneously
    would be impractical, but the anatomical processing can be done once, and
    then the functional processing can be done separately for each session.

See also the ``--level`` flag, which can be used to control which derivatives are
generated.

Troubleshooting
---------------
Logs and crashfiles are output into the
``<output dir>/fmriprep/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`Debugging Nipype Workflows <https://miykael.github.io/nipype_tutorial/notebooks/basic_debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://fmriprep.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/nipreps/fmriprep/issues.

If you have a problem or would like to ask a question about how to use *fMRIPRep*,
please submit a question to `NeuroStars.org <https://neurostars.org/tag/fmriprep>`_ with an ``fmriprep`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

Previous questions about *fMRIPRep* are available here:
https://neurostars.org/tag/fmriprep/

To participate in the *fMRIPRep* development-related discussions please use the
following mailing list: https://mail.python.org/mailman/listinfo/neuroimaging
Please add *[fmriprep]* to the subject line when posting on the mailing list.


.. include:: license.rst
