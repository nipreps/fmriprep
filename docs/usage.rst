.. include:: links.rst

Usage
-----

Execution and the BIDS format
=============================

The ``fmriprep`` workflow takes as principal input the path of the dataset
that is to be processed.
The input dataset is required to be in valid :abbr:`BIDS (Brain Imaging Data
Structure)` format, and it must include at least one T1w structural image and
(unless disabled with a flag) a BOLD series.
We highly recommend that you validate your dataset with the free, online
`BIDS Validator <http://incf.github.io/bids-validator/>`_.

The exact command to run ``fmriprep`` depends on the Installation_ method.
The common parts of the command follow the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.
Example: ::

    fmriprep data/bids_root/ out/ participant -w work/


Command-Line Arguments
======================

.. argparse::
   :ref: fmriprep.cli.run.get_parser
   :prog: fmriprep
   :nodefault:
   :nodefaultconst:


The docker wrapper CLI
======================

.. argparse::
   :ref: fmriprep_docker.get_parser
   :prog: fmriprep-docker
   :nodefault:
   :nodefaultconst:


Debugging
=========

Logs and crashfiles are outputted into the
``<output dir>/fmriprep/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

Support and communication
=========================

The documentation of this project is found here: http://fmriprep.readthedocs.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/poldracklab/fmriprep/issues.

If you have a problem or would like to ask a question about how to use ``fmriprep``,
please submit a question to `NeuroStars.org <http://neurostars.org/tags/fmriprep>`_ with an ``fmriprep`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

All previous ``fmriprep`` questions are available here:
http://neurostars.org/tags/fmriprep/

To participate in the ``fmriprep`` development-related discussions please use the
following mailing list: http://mail.python.org/mailman/listinfo/neuroimaging
Please add *[fmriprep]* to the subject line when posting on the mailing list.
