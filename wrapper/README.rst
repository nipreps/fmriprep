The fMRIPrep on Docker wrapper
------------------------------

fMRIprep is a functional magnetic resonance image pre-processing pipeline
that is designed to provide an easily accessible, state-of-the-art interface
that is robust to differences in scan acquisition protocols and that requires
minimal user input, while providing easily interpretable and comprehensive
error and output reporting.

This is a lightweight Python wrapper to run fMRIPrep.
It generates the appropriate Docker commands, providing an intuitive interface
to running the fMRIPrep workflow in a Docker environment.
Docker must be installed and running. This can be checked
running ::

  docker info

Please report any feedback to our `GitHub repository
<https://github.com/poldracklab/fmriprep>`_ and do not
forget to `credit <https://fmriprep.readthedocs.io/en/latest/citing.html>`_ all
the authors of software that fMRIPrep uses.
