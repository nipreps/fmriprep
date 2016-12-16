.. include:: links.rst

Usage
-----

Execution and the BIDS format
=============================

The ``fmriprep`` workflow takes as principal input the path of the dataset
that is to be processed.
The only requirement to the input dataset is that it has a valid BIDS_ (Brain
Imaging Data Structure) format.
This can be easily checked online using the
`BIDS Validator <http://incf.github.io/bids-validator/>`_.

The exact command to run ``fmriprep`` depends on the Installation_ method.
The common parts of the command follow the
`BIDS-Apps <https://github.com/BIDS-Apps>`_ definition.
Example: ::

    fmriprep data/bids_root/ out/ participant -w work/

Command-Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

.. highlight:: none

.. include:: args.txt
   :literal:

Debugging
=========

Logs and crashfiles are outputted into the ``<output dir>/logs`` directory.
Information on how to customize and understand these files can be found on the `nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_ page.

Support and communication
=========================

The documentation of this project is found here: http://fmriprep.readthedocs.org/en/latest/.

If you have a problem or would like to ask a question about how to use ``fmriprep``,
please submit a question to `NeuroStars.org <http://neurostars.org>`_ with an ``fmriprep`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

All previous ``fmriprep`` questions are available here:
http://neurostars.org/t/fmriprep/

To participate in the ``fmriprep`` development-related discussions please use the
following mailing list: http://mail.python.org/mailman/listinfo/neuroimaging
Please add *[fmriprep]* to the subject line when posting on the mailing list.


All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/poldracklab/fmriprep/issues.
