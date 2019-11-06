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

ERROR: it appears that ``recon-all`` is already running
-------------------------------------------------------
When running FreeSurfer_'s ``recon-all``, an error may say *it appears it is already running*.
FreeSurfer creates files (called ``IsRunning.{rh,lh,lh+rh}``, under the ``scripts/`` folder)
to determine whether it is already executing ``recon-all`` on that particular subject
in another process, compute node, etc.
If a FreeSurfer execution terminates abruptly, those files are not wiped out, and therefore,
the next time you try to execute ``recon-all``, FreeSurfer *thinks* it is still running.
The output you get from fMRIPrep will contain something like: ::

  RuntimeError: Command:
  recon-all -autorecon2-volonly -openmp 8 -subjid sub-020 -sd /outputs/freesurfer -nogcareg -nocanorm -nocareg -nonormalization2 -nomaskbfs -nosegmentation -nofill
  Standard output:
  Subject Stamp: freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a
  Current Stamp: freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a
  INFO: SUBJECTS_DIR is /outputs/freesurfer
  Actual FREESURFER_HOME /opt/freesurfer
  -rw-rw-r-- 1 11239 users 207798 Apr  1 16:19 /outputs/freesurfer/sub-020/scripts/recon-all.log
  Linux 62324c0da859 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux

  ERROR: it appears that recon-all is already running
  for sub-020 based on the presence of /outputs/freesurfer/sub-020/scripts/IsRunning.lh+rh. It could
  also be that recon-all was running at one point but
  died in an unexpected way. If it is the case that there
  is a process running, you can kill it and start over or
  just let it run. If the process has died, you should type:

  rm /outputs/freesurfer/sub-020/scripts/IsRunning.lh+rh

  and re-run. Or you can add -no-isrunning to the recon-all
  command-line. The contents of this file are:
  ----------------------------------------------------------
  ------------------------------
  SUBJECT sub-020
  HEMI    lh rh
  DATE Fri Mar 22 20:33:09 UTC 2019
  USER root
  HOST 622795a21a5f
  PROCESSID 55530
  PROCESSOR x86_64
  OS Linux
  Linux 622795a21a5f 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
  $Id: recon-all,v 1.580.2.16 2017/01/18 14:11:24 oesteban Exp $
  ----------------------------------------------------------
  Standard error:

  Return code: 1


As suggested by the ``recon-all`` output message, deleting these files will enable
FreeSurfer to execute ``recon-all`` again.
In general, please be cautious of deleting files and mindful why a file may exist.


Running subjects in parallel
----------------------------
When running several subjects in parallel, and depending on your settings, fMRIPrep may
fall into race conditions.
A symptomatic output looks like: ::

  FileNotFoundError: [Errno 2] No such file or directory: '/scratch/03201/jbwexler/openneuro_fmriprep/data/ds000003_work/ds000003-download/derivatives/fmriprep-1.4.0/fmriprep/logs/CITATION.md'

If you would like to run *fMRIPrep* in parallel on multiple subjects please use
`this method <https://neurostars.org/t/fmriprep-workaround-for-running-subjects-in-parallel/4449>`__.

How much CPU time and RAM should I allocate for a typical fMRIPrep run?
-----------------------------------------------------------------------
The recommended way to run fMRIPrep is to process one subject per container instance. A typical preprocessing run
without surface processing with freesurfer can be completed in about 2 hours with 4 CPUs or in about 1 hour with 16 CPUs.
More than 16 CPUs do not translate into faster processing times for a single subject. About 8GB of memory should be
available for a single subject preprocessing run.

Below are some benchmark data that have been computed on a high performance cluster compute node with Intel E5-2683 v4
CPUs and 64 GB of physical memory:

.. figure:: _static/fmriprep_benchmark.svg
    :scale: 100%

**Compute Time**: time in hours to complete the preprocessing for all subjects. **Physical Memory**: the maximum of RAM usage
used across all fMRIPrep processes as reported by the HCP job manager. **Virtual Memory**: the maximum of virtual memory used
across all fMRIPrep processes as reported by the HCP job manager. **Threads**: the maximum number of threads per process as
specified with ``–omp-nthreads`` in the fMRIPrep command.

The above figure illustrates that processing 2 subjects in 2 fMRIPrep instances with 8 CPUs each is approximately as fast as
processing 2 subjects in one fMRIPrep instance with 16 CPUs. However, on a distributed compute cluster, the two 8 CPU
instances may be allocated faster than the single 16 CPU instance, thus completing faster in practice. If more than one
subject is processed in a single fMRIPrep instance, then limiting the number of threads per process to roughly the
number of CPUs divided by the number of subjects is most efficient.

.. _upgrading:

A new version of *fMRIPrep* has been published, when should I upgrade?
----------------------------------------------------------------------
We follow a philosophy of releasing very often, although the pace is slowing down
with the maturation of the software.
It is very likely that your version gets outdated over the extent of your study.
If that is the case (an ongoing study), then we discourage changing versions.
In other words, **the whole dataset should be processed with the same version (and
same container build if they are being used) of *fMRIPrep*.**

On the other hand, if the project is about to start, then we strongly recommend
using the latest version of the tool.

In any case, if you can find your release listed as *flagged* in `this file
of our repo <https://github.com/poldracklab/fmriprep/blob/master/.versions.json>`__,
then please update as soon as possible.

What is *TemplateFlow* for?
---------------------------
*TemplateFlow* enables *fMRIPrep* to generate preprocessed outputs spatially normalized to
a number of different neuroimaging templates (e.g. MNI).
For further details, please check `its documentation section <spaces.html#templateflow>`__.

I'm running *fMRIPrep* via Singularity containers - how can I troubleshoot problems?
------------------------------------------------------------------------------------
We have extended `this documentation <singularity.html>`__ to cover some of the most
frequent issues other Singularity users have been faced with.
Generally, users have found it hard to `get TemplateFlow and Singularity to work
together <singularity.html#singularity-tf>`__.

Skipping the FreeSurfer `reconall` step
-----------------------------------------
We have added a flag called `--fs-no-reconall` to pass through. This will skip the FreeSurfer reconall processing. Please be aware that providng *fMRIPrep* the FreeSurfer license with the `--fs-license-path` flag is still required. 
