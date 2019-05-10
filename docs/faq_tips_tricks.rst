.. include:: links.rst

========================
FAQ, Tips, and Tricks
========================

Should I run quality control of my images before running *fMRIPrep*?
--------------------------------------------------------------------

Yes. You should do so before any processing/analysis takes place.

Oftentimes (more often than we would like), images have fatal artifacts and problems.

Some exclusion criteria for data quality should be pre-specified before QC and any screening
of the original data.
Those exclusion criteria must be designed in agreement with the goals and challenges of the
experimental design.
For instance, when it is planned to run some cortical thickness analysis, images should be excluded
even when they present the most subtle ghosts or other artifacts that may introduce biases in surface
reconstruction.
However, if the same artifactual data was planned to be used just as a reference for spatial
normalization, some of those artifacts should be noted, but may not grant exclusion of the data.

When using publicly available datasets, an additional concern is that images may have gone through
some kind of preprocessing (see next question).


What if I find some images have undergone some pre-processing already (e.g., my T1w image is already skull-stripped)?
---------------------------------------------------------------------------------------------------------------------

These images imply an unknown level of preprocessing (e.g. was it already bias-field corrected?),
which makes it difficult to decide on best-practices for further processing.
Hence, supporting such images was considered very low priority for *fMRIPrep*.
For example, see `#707 <https://github.com/poldracklab/smriprep/issues/12>`_ and an illustration of
downstream consequences in `#939 <https://github.com/poldracklab/fmriprep/issues/939>`_.

So for OpenFMRI, we've been excluding these subjects, and for user-supplied data, we would recommend
reverting to the original, defaced, T1w images to ensure more uniform preprocessing.


My *fMRIPrep* run is hanging...
-------------------------------

When running on Linux platforms (or containerized environments, because they are built around
Ubuntu), there is a Python bug that affects *fMRIPrep* that drives the Linux kernel to kill
processes as a response to running out of memory.
Depending on the process killed by the kernel, *fMRIPrep* may crash with a ``BrokenProcessPool``
error or hang indefinitely, depending on settings.
While we are working on finding a solution that does not run up against this bug, this may take some
time.
This can be most easily resolved by allocating more memory to the process, if possible.

Please find more information regarding this error from discussions on
`NeuroStars <https://neurostars.org/tags/fmriprep>`_:

 * `memory issue when processing large amount of data <https://neurostars.org/t/memory-issue-when-processing-large-amount-of-data/2562>`_
 * `RAM CPUs reasonable to run pipelines like fMRIPrep <https://neurostars.org/t/how-much-ram-cpus-is-reasonable-to-run-pipelines-like-fmriprep/1086>`_
 * `memory allocation issues with fMRIPrep, Singularity and HPC <https://neurostars.org/t/memory-allocation-issues-with-fmriprep-singularity-and-hpc/2759>`_
 * `fMRIPrep v1.0.12 hanging <https://neurostars.org/t/fmriprep-v1-0-12-hanging/1661>`_.

Additionally, consider using the ``--low-mem`` flag, which will make some memory optimizations at the cost of disk space in the working directory.
