.. include:: links.rst

================================
FAQ - Frequently Asked Questions
================================

.. contents::
    :local:
    :depth: 1

------------

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
For example, see `#707 <https://github.com/nipreps/smriprep/issues/12>`_ and an illustration of
downstream consequences in `#939 <https://github.com/nipreps/fmriprep/issues/939>`_.

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

I have already run ``recon-all`` on my subjects, can I reuse my outputs?
------------------------------------------------------------------------
Yes, as long as the FreeSurfer_ version previously used was ``6.0.0`` or newer.
If running with FreeSurfer, *fMRIPrep* checks if the output directory contains a ``freesurfer``
directory and reuses the outputs found.
Alternatively, you can use the ``--fs-subjects-dir`` flag to specify a different location for the existing FreeSurfer outputs.

ERROR: it appears that ``recon-all`` is already running
-------------------------------------------------------
When running FreeSurfer's ``recon-all``, an error may say *it appears it is already running*.
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
When running several subjects in parallel, and depending on your settings, *fMRIPrep* may
fall into race conditions.
A symptomatic output looks like: ::

  FileNotFoundError: [Errno 2] No such file or directory: '/scratch/03201/jbwexler/openneuro_fmriprep/data/ds000003_work/ds000003-download/derivatives/fmriprep-1.4.0/fmriprep/logs/CITATION.md'

If you would like to run *fMRIPrep* in parallel on multiple subjects please use
`this method <https://neurostars.org/t/updated-fmriprep-workaround-for-running-subjects-in-parallel/6677>`__.

How much CPU time and RAM should I allocate for a typical *fMRIPrep* run?
-------------------------------------------------------------------------
The recommended way to run *fMRIPrep* is to process one subject per container instance. A typical preprocessing run
without surface processing with FreeSurfer can be completed in about 2 hours with 4 CPUs or in about 1 hour with 16 CPUs.
More than 16 CPUs do not translate into faster processing times for a single subject. About 8GB of memory should be
available for a single subject preprocessing run.

Below are some benchmark data that have been computed on a high performance cluster compute node with Intel E5-2683 v4
CPUs and 64 GB of physical memory:

.. figure:: _static/fmriprep_benchmark.svg

**Compute Time**: time in hours to complete the preprocessing for all subjects. **Physical Memory**: the maximum of RAM usage
used across all *fMRIPrep* processes as reported by the HCP job manager. **Virtual Memory**: the maximum of virtual memory used
across all *fMRIPrep* processes as reported by the HCP job manager. **Threads**: the maximum number of threads per process as
specified with ``–omp-nthreads`` in the fMRIPrep command.

The above figure illustrates that processing 2 subjects in 2 *fMRIPrep* instances with 8 CPUs each is approximately as fast as
processing 2 subjects in one *fMRIPrep* instance with 16 CPUs. However, on a distributed compute cluster, the two 8 CPU
instances may be allocated faster than the single 16 CPU instance, thus completing faster in practice. If more than one
subject is processed in a single *fMRIPrep* instance, then limiting the number of threads per process to roughly the
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
of our repo <https://github.com/nipreps/fmriprep/blob/master/.versions.json>`__,
then please update as soon as possible.

I'm running *fMRIPrep* via Singularity containers - how can I troubleshoot problems?
------------------------------------------------------------------------------------
We have extended `this documentation <singularity.html>`__ to cover some of the most
frequent issues other Singularity users have been faced with.
Generally, users have found it hard to `get TemplateFlow and Singularity to work
together <singularity.html#singularity-tf>`__.

What is *TemplateFlow* for?
---------------------------
*TemplateFlow* enables *fMRIPrep* to generate preprocessed outputs spatially normalized to
a number of different neuroimaging templates (e.g. MNI).
For further details, please check `its documentation section <spaces.html#templateflow>`__.

.. _tf_no_internet:

How do you use *TemplateFlow* in the absence of access to the Internet?
-----------------------------------------------------------------------
This is a fairly common situation in :abbr:`HPCs (high-performance computing)`
systems, where the so-called login nodes have access to the Internet but
compute nodes are isolated, or in PC/laptop environments if you are traveling.
*TemplateFlow* will require Internet access the first time it receives a
query for a template resource that has not been previously accessed.
If you know what are the templates you are planning to use, you could
prefetch them using the Python client.
In addition to the ``--output-spaces`` that you specify, *fMRIPrep* will
internally require the ``MNI152NLin2009cAsym`` template.
If the ``--skull-strip-template`` option is not set, then ``OASIS30ANTs``
will be used.
Finally, the ``--cifti-output`` argument requires ``MNI152NLin6Asym``.
To do so, follow the next steps.

1. By default, a mirror of *TemplateFlow* to store the resources will be
   created in ``$HOME/.cache/templateflow``.
   You can modify such a configuration with the ``TEMPLATEFLOW_HOME``
   environment variable, e.g.::

     $ export TEMPLATEFLOW_HOME=$HOME/.templateflow

2. Install the client within your favorite Python 3 environment (this can
   be done in your login-node, or in a host with Internet access,
   without need for Docker/Singularity)::

     $ python -m pip install -U templateflow

3. Use the ``get()`` utility of the client to pull down all the templates you'll
   want to use. For example::

     $ python -c "from templateflow.api import get; get(['MNI152NLin2009cAsym', 'MNI152NLin6Asym', 'OASIS30ANTs', 'MNIPediatricAsym', 'MNIInfant'])"

After getting the resources you'll need, you will just need to make sure your
runtime environment is able to access the filesystem, at the location of your
*TemplateFlow home* directory.
If you are an Apptainer (formerly Singularity) user, please check out `TemplateFlow and Singularity
<https://www.nipreps.org/apps/singularity/#templateflow-and-singularity>`__.

How do I select only certain files to be input to *fMRIPrep*?
-------------------------------------------------------------
Using the ``--bids-filter-file`` flag, you can pass *fMRIPrep* a JSON file that
describes a custom BIDS filter for selecting files with PyBIDS, with the syntax
``{<query>: {<entity>: <filter>, ...},...}``. For example::

  {
      "t1w": {
          "datatype": "anat",
          "session": "02",
          "acquisition": null,
          "suffix": "T1w"
      },
      "bold": {
          "datatype": "func",
          "session": "02",
          "suffix": "bold"
      }
  }

fMRIPrep uses the following queries, by default::

  {
    "fmap": {"datatype": "fmap"},
    "bold": {"datatype": "func", "suffix": "bold"},
    "sbref": {"datatype": "func", "suffix": "sbref"},
    "flair": {"datatype": "anat", "suffix": "FLAIR"},
    "t2w": {"datatype": "anat", "suffix": "T2w"},
    "t1w": {"datatype": "anat", "suffix": "T1w"},
    "roi": {"datatype": "anat", "suffix": "roi"}
  }

Only modifications of these queries will have any effect. You may filter on any entity defined
in the PyBIDS
`config file <https://github.com/bids-standard/pybids/blob/master/bids/layout/config/bids.json>`__.
To select images that do not have the entity set, use json value: ``null``.
To select images that have any non-empty value for an entity use string: ``'*'``

Can *fMRIPrep* continue to run after encountering an error?
-----------------------------------------------------------
(Context: `#1756 <https://github.com/nipreps/fmriprep/issues/1756>`__)
Yes, although it requires access to previously computed intermediate results.
*fMRIPrep* is built on top of Nipype_, which uses a temporary folder to store the interim
results of the workflow.
*fMRIPrep* provides the ``-w <PATH>`` command line argument to set a customized temporal
folder (the *working directory*, in the following) for the *Nipype* workflow engine.
By default, *fMRIPrep* configures the *working directory* to be ``$PWD/work/``.
Therefore, if your *fMRIPrep* process crashes and you attempt to re-run it reusing
as much as it could from the previous run, you can either make sure that
the default ``$PWD/work/`` points to a reasonable, reusable path in your environment or
configure a better location on your with ``-w <PATH>``.

Can I use *fMRIPrep* for longitudinal studies?
----------------------------------------------
As partially indicated before, *fMRIPrep* assumes no substantial anatomical changes happen
across sessions.
When substantial changes are expected, special considerations must be taken.
Some examples follow:

* Surgery: use only pre-operation sessions for the anatomical data. This will typically be done
  by omitting post-operation sessions from the inputs to *fMRIPrep*.
* Developing and elderly populations: there is currently no standard way of processing these.
  However, `as suggested by U. Tooley at NeuroStars.org
  <https://neurostars.org/t/fmriprep-how-to-reuse-longitudinal-and-pre-run-freesurfer/4585/15>`__,
  it is theoretically possible to leverage the *anatomical fast-track* along with the
  ``--bids-filters`` option to process sessions fully independently, or grouped by some study-design
  criteria.
  Please check the `link
  <https://neurostars.org/t/fmriprep-how-to-reuse-longitudinal-and-pre-run-freesurfer/4585/15>`__
  for further information on this approach.


How to decrease *fMRIPrep* runtime when working with large datasets?
--------------------------------------------------------------------
*fMRIPrep* leverages PyBIDS to produce a layout, which indexes the input BIDS dataset and facilitates file queries.
Depending on the amount of files and metadata within the BIDS dataset, this process can be time-intensive.
As of the 20.2.0 release, *fMRIPrep* supports the ``--bids-database-dir <database_dir>`` option,
which can be used to pass in an already indexed BIDS layout.

The default *fMRIPrep* layout can be generated by running the following shell command (requires PyBIDS 0.13.2 or greater)::

  pybids layout <bids_root> <database_dir> --no-validate --index-metadata

where ``<bids_root>`` indicates the root path of the BIDS dataset, and ``<database_dir>``
is the path where the pre-indexed layout is created - which is then passed into *fMRIPrep*.

By using the ``--force-index`` and ``--ignore`` options,
finer control can be achieved of what files are visible to fMRIPrep.

Note that any discrepancies between the pre-indexed database and
the BIDS dataset complicate the provenance of fMRIPrep derivatives.
If ``--bids-database-dir`` is used, the referenced directory should be
preserved for the sake of reporting and reproducibility.

Error in slice timing correction: *insufficient length of BOLD data after discarding nonsteady-states*
------------------------------------------------------------------------------------------------------
Typically, the scanner will be in a *nonsteady state* during a few initial time points of the acquisition,
until it stabilizes.
These *nonsteady states* (also called *dummy* scans) typically show greater T1 contrast and higher average
intensity, and therefore potentially are detrimental if used in the interpolation of slice timing corrections.
Hence, *nonsteady states* are discarded by the slice timing correction tool (in this case, AFNI's ``3dTShift``).
However, ``3dTShift`` requires that at least five (5) time points are present in the target series, after
dismissing the initial *nonsteady states*.

*fMRIPrep* estimates the number of *nonsteady states* within the pipeline, unless the parameter is provided
by the user with the argument ``--dummy-scans <num>``.
Either way, if the number of *nonsteady states* is, say 4, then the length of the BOLD series must be greater
than 8.
If you encounter this error, first check that the number of *nonsteady states* is not suspiciously large
(it typically ranges from zero to five).
Next, if the number of *nonsteady states* is reasonable, consider why your BOLD time series are so short
and whether slice timing correction is appropriate under these conditions.
Finally, you can either skip the slice-timing correction with the argument ``--ignore slicetiming`` or
enforce a number of *nonsteady states* lower than the maximum for your data with ``--dummy-scans <num>``.
Please note that both strategies will apply to all tasks and runs that are to be processed.
