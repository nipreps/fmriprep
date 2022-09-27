22.0.2 (September 27, 2022)
===========================
A patch release in the 22.0.x series.

This release increases the minimum Nipype version to include better error messages on failures.
Additionally, this includes a fix to allow SyN distortion correction in combination with the
`--ignore fieldmaps` option.

  * MAINT: Add `pre-commit`, dev installation for consistent styling (#2857)
  * CI: Upgrade docker orb (#2858)

22.0.1 (September 13, 2022)
===========================
Patch release in the 22.0.x series.

This release includes relaxed tolerance in the case where fieldmap affines slightly differed,
and a fix for running FreeSurfer 7 with an outdated fsaverage folder.

  * ENH: Add check to ensure latest fsaverage folder is used (#2847)
  * FIX: Filter sbrefs by BIDS filters if available (#2843)
  * FIX: Avoid crashing on empty ACompCor results (#2850)

22.0.0 (July 28, 2022)
======================
New feature release in the 22.0.x series.

This release has been tested to work with Python 3.9 and FreeSurfer 7.2,
which are now bundled in the Docker image.

This release also features improvements to T2\* and carpetplot reporting.

For the next release, we are investigating issues with susceptibility
distortion correction (SDC). Please check your results carefully and report
any issues you find.

  * FIX: Update wrapper python path (#2783)
  * FIX: Preserve ``collect_data`` behavior by using named args (#2754)
  * FIX: Update *fMRIPrep* version in bibliography at run time (#2738)
  * FIX: Plot carpetplot with CIFTI-specific colorbar (#2737)
  * FIX: Two minor typos in report spec (#2708)
  * ENH: Clip T2\* values at 100ms to keep consistent histogram axes (#2781)
  * ENH: Plot histogram of T2\* values in gray-matter mask (#2778)
  * ENH: Save T2starmap files in all requested output spaces, if calculated (#2776)
  * ENH: Compare T2\* map to BOLD reference (#2751)
  * ENH: Add edge-regressors to confounds & crown to carpetplot (#2621)
  * ENH: Add major/minor versions to base workflow name (#2716)
  * DOC: Fix JSON typo in config in FAQ (#2771)
  * DOC: Clarify calculation of confounding signals (#2724)
  * MNT: Build on Python 3.9 environment (#2782)
  * MNT: Seed ignore-revs file and script to tag new hashes with log entries (#2748)
  * MNT: Require PyBIDS 0.15+ to allow fMRIPrep to preserve zero-padding in run entity (#2745)
  * MNT: Upload artifacts after each step of Circle's workflow (#2736)
  * MNT: Normalize code style of ``workflows.confounds`` (#2729)
  * MNT: Ask for fmriprep-docker RUNNING line (#2670)
  * DOCKER: Bundle FreeSurfer 7 (#2779)
  * CI: Touch up CircleCI configuration (#2764)
  * CI: Update package builds to use python -m build (#2746)

21.0.2 (April 21, 2022)
=======================
Bug-fix releaase in the 21.0.x series.

This release includes a few bug-fixes for susceptibility distortion correction (SDC) and multi-echo (ME).
A few notable fixes include:
- Added tolerance for affine precision differences when using EPI fieldmaps.
- Removed hang-ups when reusing anatomical derivatives with ME data.
- Increased BOLD masking workflow robustness.

A full list of changes can be found below:

  * DOCKER: Update multiarch deb package link (#2758)
  * ENH: Add affine_tolerance flag to MergeSeries (nipreps/niworkflows#706)
  * FIX: Initialize BIDS layout after cleaning working directory (#2741)
  * FIX: Avoid double unwarping during resampling of processed multi-echo data (#2730)
  * FIX: Account for potential lists of lists in multi-echo cases (nipreps/niworkflows#719)
  * FIX: Improve reliability of BOLD masking workflow (nipreps/niworkflows#712)
  * FIX: Relax tolerance for different affines when concatenating blips (nipreps/sdcflows#265)

21.0.1 (January 24, 2022)
=========================
Bug-fix release in the 21.0.x series.

This release re-enables reverse-PE EPI fieldmaps, i.e. EPIs acquired with the
opposite phase-encoding direction to a BOLD series.
Additionally, the ``--use-syn-sdc`` option will now error by default if no
``PhaseEncodingDirection`` can be found for a BOLD series. To reduce this to
a warning, use ``--use-syn-sdc warn``. SyN-SDC will NOT be run, but the workflow
will proceed.

.. note::

  ``topup`` runtime scales with the number of volumes used, so fMRIPrep defaults to
  using a maximum of 5 volumes from each of the BOLD series and the EPI fieldmap.
  To adjust this, please use the ``--topup-max-vols`` parameter.

* FIX: Create single fmap_select_std node for use with all ANAT estimators (#2692)
* FIX: Re-enable reverse-PE EPI fieldmaps (#2684)
* FIX: Pass boolean use_syn_sdc to find_estimators (#2685)
* FIX: Ensure AROMA inputs are uncompressed if --low-mem (#2657)
* ENH: Add ``--topup-max-vols`` flag to control TOPUP runtime (#2688)
* ENH: Allow ``--use-syn-sdc`` to take a "warn" option to avoid exiting when PE dir is unavailable (#2680)
* RF: Add extension field to differentiate from upcoming templates (#2613)
* WRAPPER: Map ``--bids-filter-file``\s into Docker container (#2691)
* MNT: Silence warning from pybids; list packaging as dependency (#2689)
* MNT: Add netbase to Docker image to ensure correct datalad operation (#2682)
* MNT: Restore nitime to requirements, necessary for DVARS (#2678)

21.0.0 (December 14, 2021)
==========================
A new series is finally here!

Some highlights of this version's new features include a substantial rework of fMRIPrep's
susceptibility distortion correction (including the addition of FSL's ``topup`` as the new
default PEPOLAR correction technique), improved flexibility for multiecho scans, and
dependency version upgrades within the Docker environment.

.. note::

  To follow BIDS Derivatives more closely, *fMRIPrep*'s default output layout has changed.
  To enable the legacy layout, use the ``--output-layout legacy`` flag.
  For more information, please see https://fmriprep.org/en/latest/outputs.html#layout

.. caution::

    As with all minor version increments, working directories
    from previous versions **should not be reused**.

Thank you for using *fMRIPrep*!
If you encounter any issues with this release, please let us know
by posting an issue on our GitHub page!

A full list of changes can be found below.

* DOC: Remove mention to ``epidewarp.fsl`` from ``NOTICE`` (#2629)
* DOC: Update description of output layout, add discussion of legacy mode (#2646)
* DOC: ME | Add missing output to documentation, improve boilerplate (#2608)
* DOC: Add --index-metadata to ``pybids layout`` faq (#2546)
* DOC: Add warning about slice timing correction in output documentation. (#2502)
* DOC: Update sample report (#2519)
* DOC: Refactor README and citing information (#2474)
* DOC: Point documentation at *SDCFlows* and remove SDC section (#2470)\
* DOC: Transfer duplicated documentation to www.nipreps.org (#2469)
* DOC: Better explanation on how spike regressors are generated (#2465)
* DOC: Clarify that ``res-2`` entity does not mean 2mm (#2466)
* DOC: adds --mem metavar (#2378)
* DOC: skull stripping is forced by default (#2345)
* DOCKER: Shrink Docker image size (#2551)
* DOCKER: Strip ABI tag from libQt5Core.so.5 to prevent Singularity failures (#2535)
* ENH: Make bids the default output layout option (#2555)
* ENH: Mask multi-echo data with reference mask instead of echo-specific masks (#2349)
* ENH: Replace initial FLIRT with mri_coreg, use -basescale 1 for FLIRT-BBR (#2625)
* ENH: Add flag for STC reference time and set in all cases (#2520)
* ENH: Use ``BIDSLayoutIndexer`` and do not index unnecessary modalities (#2494)
* ENH: Allow users to download templateflow templates to directory (#2482)
* ENH: Detect 3D & too-short BOLD series, warn and skip run's workflow building (#2461)
* ENH: Modernize Dockerfile (#2409) (#2270)
* ENH: Leverage SDCFlows 2.x API (#2392)
* ENH: Tedana-friendly ME derivatives (#2575)
* ENH: Better integration of *SDCFlows*' unwarping (#2576)
* ENH: Refactor workflow to adopt *SDCFlows*' new correction workflow (#2547)
* FIX: Threshold boldref resampled values to be non-negative (#2630)
* FIX: Restore SyN-SDC (#2530)
* FIX: Confusing I/O field ``bold_mask`` in bold-to-bold resampling workflow (#2611)
* FIX: ME | Revise SDC compute graph (#2610)
* FIX: Propagate the SDC warp to resampling node also with ME (#2609)
* FIX: Optimize build job on CircleCI (#2605)
* FIX: Minor multiecho issues (#2574)
* FIX: Missing numpy import (#2533)
* FIX: Avoid eager imports from *NiWorkflows* causing API breaks (#2495)
* FIX: Split aCompCor into aCompCor, cCompCor, wCompCor (#2523)
* FIX: Avoid unnecessary connections based on branching logic (#2508)
* FIX: Permit missing TR to show PyBIDS error at workflow construction time (#2513)
* FIX: Simplify STC logic for too short BOLD series (#2489)
* FIX: Catch FreeSurfer error related to FIPS being enabled (#2490)
* FIX: ``DerivativesDataSink`` nondeterministic checksums fixed with ``niworkflows=~1.3.4`` (#2458)
* FIX: Address dependency incompatibilities by pinning specific packages (#2463)
* FIX: Unprotected import of ``sentry_sdk``, which is not a dependency (#2460)
* FIX: Clarify phase encoding direction, rather than axis (#2302)
* MAINT: Conform fast-track outputs to BIDS ordering (#2663)
* MAINT: Reduce redundant processing of ds005 (#2649)
* MAINT: Finalize the work we started with the refactor of the Docker image (#2601)
* MAINT: Run full workflows on tags (#2593)
* MAINT: Stage Python environment of Docker image from nipreps/miniconda (#2581)
* MAINT: Stop NeuroDebian dependency in Dockerfile (#2578)
* MAINT: Run pytest through GitHub actions (#2529)
* MAINT: Relicense +20.3.x - BSD-3-Clause -> Apache License 2.0 (#2325)
* MAINT: Add missing OASIS30 WM/BS probsegs (#2471)
* MAINT: Update BIDS validator to 1.8.0 (#2443)
* MAINT: Address the problems of a sloppy merge (#2468)
* MAINT: Containers - remove ``/root/.npm`` (#2464)
* MAINT: Remove traces of Xenial base (#2459)
* MAINT: Revise CircleCI's cache tags to avoid mixups between builds (#2457)
* MAINT: Back port CircleCI configuration from dev branch (#2456)
* MAINT: Increase minimum tedana version (#2366)
* MAINT: Docker hub has disabled password authentication (#2448)
* MAINT: Add Lea Waller to ``contributors.json`` (#2304)
* MAINT: Move away from nearly deprecated images (#2295)
* MAINT: Migrate from poldracklab to nipreps orgs (#2290)
* MAINT: Ease CI packaging tests (#2472)
* RF/FIX: Iterate over echo indices, not filenames, simplifying iteration logic (#2651)

20.2.7 (January 24, 2022)
=========================
Bug-fix release in the 20.2.x LTS series.

  * FIX: Clarify phase encoding direction, rather than axis (#2690)
  * FIX: Ensure AROMA inputs are uncompressed if --low-mem (#2657)
  * FIX: Add ``-basescale 1`` parameter to avoid ``flirt`` scaling (#2624)
  * WRAPPER: Map ``--bids-filter-file``\s into Docker container (#2691)
  * MNT: Add netbase to Docker image to ensure correct datalad operation (#2682)

20.2.6 (October 27, 2021)
=========================
Patch release in the 20.2.x LTS series.

This release includes an updated runtime environment for Docker/Singularity users.
This environment includes a newer release of ``indexed_gzip``,
fixing ``CrcError``\s previously encountered.

  * CI: Run full workflows on tags (#2593)
  * RF: Add extension field to differentiate from upcoming templates (#2613)

20.2.5 (October 12, 2021)
=========================
Bug-fix release in the 20.2.x LTS series.

This release includes a fix to ``--slice-time-ref`` parsing. Also, some
issues in anatomical processing are resolved, including poorly-interpolated
labels in aseg segmentations, and probabilistic segmentations have reverted
to FAST.

  * FIX: --slice-time-ref option parsing (#2573)
  * CI: Add style checks (missing since Travis got throttled) (#2570)

20.2.4 (October 04, 2021)
=========================
Bug-fix release in the 20.2.x LTS series.

This release includes some fixes to minor bugs, and includes enhancements
to allow the user to specify a different slice-timing-correction (STC) target.

Each BOLD series will contain ``SliceTimingCorrected`` metadata to indicate
whether STC was performed. If performed, the ``StartTime`` metadata field will
be set to indicate the onset time of the BOLD series.

  * FIX: Avoid unnecessary connections based on branching logic (#2508)
  * FIX: Permit missing TR to show PyBIDS error at workflow construction time (#2513)
  * [BACKPORT] FIX: Catch FreeSurfer error related to FIPS being enabled (#2490)
  * ENH: Use ``BIDSLayoutIndexer`` and do not index unnecessary modalities (#2494)
  * [BACKPORT] ENH: Slice-timing correction improvements (#2565)

20.2.3 (July 21, 2021)
======================
Patch release in the 20.2.x LTS series.
Addresses minimal problems in some edge-cases, overall improves the documentation and infrastructure.
Exercises some maintenance operations addressing problems surfaced when the previous tag 20.2.2 was released.

  * FIX: Address the problems of a sloppy merge (#2468)
  * FIX: ``DerivativesDataSink`` nondeterministic checksums fixed with ``niworkflows=~1.3.4`` (#2458)
  * FIX: Address dependency incompatibilities by pinning specific packages (#2463)
  * FIX: Unprotected import of ``sentry_sdk``, which is not a dependency (#2460)
  * ENH: Detect 3D & too-short BOLD series, warn and skip run's workflow building (#2461)
  * DOC: Transfer duplicated documentation to www.nipreps.org (#2469)
  * DOC: Better explanation on how spike regressors are generated (#2465)
  * DOC: Clarify that ``res-2`` entity does not mean 2mm (#2466)
  * MAINT: Containers - remove ``/root/.npm`` (#2464)
  * MAINT: Back port CircleCI configuration from dev branch (#2456)

20.2.2 (July 16, 2021)
======================
Hot-fix release in the 20.2.x LTS series.
Addressing a bug on *fMRIPrep*'s resampling to standard spaces (#2444).
In addition, some other relevant bug-fixes and performance improvements are built in this patch.

* FIX: Feed *NiTransforms* with LTAs of type RAS2RAS (#2444)
* FIX: Add some clarity to ``BrokenProcessPool`` failures (#2436)
* FIX: Fall-back to initializing workflow in main process (#2435)
* FIX: Raise informative error when duplicate subworkflows are added (#2434)
* FIX: Non-existing path or JSON syntax error for ``--bids-filter-file`` should raise on error (#2331)
* FIX: Ignore SBRef files if ``--ignore sbref`` is passed (#2370)
* ENH: Relax requirement for *PyBIDS* databases to exist (#2429)
* ENH: Improve ``_get_series_len`` performance (#2406)
* ENH: Set and track NumPy's random seed (#2400)
* DOC: Skull-stripping is forced by default (#2430)
* MAINT: Pin ``nilearn==0.6.2`` (#2427)
* MAINT: Pin ``tedana==0.0.9a1`` for LTS branch (#2403)
* MAINT: Failing CI (#2401)

20.2.1 (November 06, 2020)
==========================
Bug-fix release in the 20.2.x LTS series.

This release adds missing metadata to GIFTI and CIFTI-2 derivatives and fixes a logging
bug affecting a small number of datasets.

Additionally the ``--output-layout bids`` flag will now output fMRIPrep derivatives directly
into the output directory, nesting FreeSurfer derivatives by default in
``<output>/sourcedata/freesurfer``. This layout allows the output root to be a valid BIDS
Derivatives dataset, simplifying data management and provenance tracking with content tracking
tools, such as `DataLad <https://handbook.datalad.org/>`__. This layout will likely become the
default in future release series.

* ENH: Output TaskName and timing metadata for all resampled BOLD series (#2320)
* ENH: Add ``--output-layout`` CLI option to enable BIDS (YODA) mode (#2303)
* ENH: Add Docker authentication to increase pull rate limit (#2316)
* FIX: Specify logger for warning (#2298)

20.2.0 (September 28, 2020)
===========================
With this third minor release series of 2020,
the first *fMRIPrep LTS* (*long-term support*) is finally here!

This release contains a number of bug-fixes and enhancements mostly
related to easing the maintenance, anticipating patch-release breaking
changes to ensure a longstanding LTS, and addressing some run-to-run
repeatability problems of the CompCor implementation.

.. admonition:: Long-Term Support (LTS)

    *fMRIPrep* 20.2 LTS introduces the `long-term support program
    <https://www.nipreps.org/devs/releases/#long-term-support-series>`__.
    This LTS version will be kindly steered and maintained by
    the group of Dr. Basile Pinsard and Prof. Pierre Bellec at
    `CRIUGM <http://www.criugm.qc.ca/>`__, (Université de Montréal).
    The LTS is planned for a window of 4 years of support (i.e., until
    September 2024).

.. caution::

    As with all minor version increments, working directories
    from previous versions **should not be reused**.

Thank you for using *fMRIPrep*!
If you encounter any issues with this release, please let us know
by posting an issue on our GitHub page!

A full list of changes can be found below.

* FIX: Pin *NiWorkflows* 1.3.1 and *sMRIPrep* 0.7.0rc2, including bugfix for INU-correction failures (nipreps/niworkflows#567)
* FIX: Get missing ``probseg`` file from MNI152NLin2009cAsym (#2271)
* FIX: Restore ``--ignore t2w/flair`` options (#2260)
* FIX: Revise the reproducibility of *CompCor* masks (#2130)
* FIX: Simplify transform aggregation in resampling, pass identity transforms for multi-echo cases (#2239)
* FIX: Skip the T1w check if ``--anat-derivatives`` is provided. (#2201)
* FIX: Storing ``--bids-filters`` within config file (#2177)
* FIX: Revise multi-echo reference generation, permitting using SBRefs too (#1803)
* FIX: *FreeSurfer* license manipulation & canary
* ENH: Output CompCor masks if ``--debug compcor`` is passed (#2248)
* ENH: Conform to BIDS Derivatives as of BIDS 1.4.0 (#2223)
* ENH: Reuse config (#2240)
* ENH: Save BOLD-anatomical transforms to derivatives folder (#2233)
* ENH: Leverage BIDSLayout's ``database_path`` (#2203)
* ENH: Add ``--no-tty`` option to ``fmriprep-docker.py`` (#2204)
* ENH: Report number of echoes in BOLD summary. (#2184)
* ENH: Ensure *NiPype* telemetry is just pinged once (#2168)
* DOC: Add FAQ entry for using pre-indexed layouts (#2256)
* DOC: Update reference in "Refinement of Brain Mask" description (#2215)
* DOC: List *TemplateFlow* templates that need to be prefetched (#2196)
* DOC: Update references to https://github.com/nipreps (#2191)
* DOC: Pin *NiPype* with new Sphinx extension syntax (#2092)
* MAINT: Upgrade ANTs to 2.3.3 in ``Dockerfile`` (9512086)
* MAINT: Track #2269 and #2269, bug-fixes on the 20.1.x series
* MAINT: Remove derivatives from layout index ignores (#2258)
* MAINT: Track #2252 from 20.1.x series (#2253)
* MAINT: Silence *PyBIDS* warning by setting extension mode (#2250)
* MAINT: Drop CircleCI docs build (#2247)
* MAINT: Pin latest *NiPreps* (#2244)
* MAINT: Update ``setup.cfg`` (flake8 and pytest) (#2183)
* MAINT: Delete release-drafter (#2169)
* MAINT: Track bug-fix release on the 20.1.x series (#2165)
* MAINT: Remove auto-comment bot (#2166)
* MAINT: Improve the questions on the bug-report template (#2158)

.. admonition:: Author list for papers based on *fMRIPrep* 20.2 LTS series

    As described in the `Contributor Guidelines
    <https://www.nipreps.org/community/CONTRIBUTING/#recognizing-contributions>`__,
    anyone listed as developer or contributor may write and submit manuscripts
    about *fMRIPrep*.
    To do so, please move the author(s) name(s) to the front of the following list:

    Markiewicz, Christopher J. \ :sup:`1`\ ; Goncalves, Mathias \ :sup:`1`\ ; DuPre, Elizabeth \ :sup:`2`\ ; Kent, James D. \ :sup:`3`\ ; Salo, Taylor \ :sup:`4`\ ; Ciric, Rastko \ :sup:`1`\ ; Pinsard, Basile \ :sup:`5`\ ; Finc, Karolina \ :sup:`6`\ ; de la Vega, Alejandro \ :sup:`7`\ ; Feingold, Franklin \ :sup:`1`\ ; Tooley, Ursula A. \ :sup:`8`\ ; Benson, Noah C. \ :sup:`9`\ ; Urchs, Sebastian \ :sup:`2`\ ; Blair, Ross W. \ :sup:`1`\ ; Erramuzpe, Asier \ :sup:`10`\ ; Lurie, Daniel J. \ :sup:`11`\ ; Heinsfeld, Anibal S. \ :sup:`12`\ ; Jacoby, Nir \ :sup:`13`\ ; Jamison, Keith W. \ :sup:`14`\ ; Frederick, Blaise B. \ :sup:`15, 16`\ ; Valabregue, Romain \ :sup:`17`\ ; Sneve, Markus H. \ :sup:`18`\ ; Liem, Franz \ :sup:`19`\ ; Adebimpe, Azeez \ :sup:`20`\ ; Velasco, Pablo \ :sup:`21`\ ; Wexler, Joseph B. \ :sup:`1`\ ; Groen, Iris I. A. \ :sup:`22`\ ; Ma, Feilong \ :sup:`23`\ ; Amlien, Inge K. \ :sup:`18`\ ; Bellec, Pierre \ :sup:`5`\ ; Cieslak, Matthew \ :sup:`20`\ ; Devenyi, Grabriel A. \ :sup:`24`\ ; Ghosh, Satrajit S. \ :sup:`25, 26`\ ; Gomez, Daniel E. P. \ :sup:`27`\ ; Halchenko, Yaroslav O. \ :sup:`23`\ ; Isik, Ayse Ilkay \ :sup:`28`\ ; Moodie, Craig A. \ :sup:`1`\ ; Naveau, Mikaël \ :sup:`29`\ ; Rivera-Dompenciel, Adriana \ :sup:`3`\ ; Satterthwaite, Theodore D. \ :sup:`20`\ ; Sitek, Kevin R. \ :sup:`30`\ ; Stojić, Hrvoje \ :sup:`31`\ ; Thompson, William H. \ :sup:`1`\ ; Wright, Jessey \ :sup:`1`\ ; Ye, Zhifang \ :sup:`32`\ ; Gorgolewski, Krzysztof J. \ :sup:`1`\ ; Poldrack, Russell A. \ :sup:`1`\ ; Esteban, Oscar \ :sup:`33`\ .

    Affiliations:

      1. Department of Psychology, Stanford University
      2. Montreal Neurological Institute, McGill University
      3. Neuroscience Program, University of Iowa
      4. Department of Psychology, Florida International University
      5. SIMEXP Lab, CRIUGM, University of Montréal, Montréal, Canada
      6. Centre for Modern Interdisciplinary Technologies, Nicolaus Copernicus University in Toruń
      7. University of Texas at Austin
      8. Department of Neuroscience, University of Pennsylvania, PA, USA
      9. Department of Psychology, New York University
      10. Computational Neuroimaging Lab, BioCruces Health Research Institute
      11. Department of Psychology, University of California, Berkeley
      12. Child Mind Institute
      13. Department of Psychology, Columbia University
      14. Department of Radiology, Weill Cornell Medicine
      15. McLean Hospital Brain Imaging Center, MA, USA
      16. Consolidated Department of Psychiatry, Harvard Medical School, MA, USA
      17. CENIR, INSERM U1127, CNRS UMR 7225, UPMC Univ Paris 06 UMR S 1127, Institut du Cerveau et de la Moelle épinière, ICM, F-75013, Paris, France
      18. Center for Lifespan Changes in Brain and Cognition, University of Oslo
      19. URPP Dynamics of Healthy Aging, University of Zurich
      20. Perelman School of Medicine, University of Pennsylvania, PA, USA
      21. Center for Brain Imaging, New York University
      22. Department of Psychology, New York University, NY, USA
      23. Dartmouth College: Hanover, NH, United States
      24. Department of Psychiatry, McGill University
      25. McGovern Institute for Brain Research, MIT, MA, USA
      26. Department of Otolaryngology, Harvard Medical School, MA, USA
      27. Donders Institute for Brain, Cognition and Behaviour, Radboud University Nijmegen
      28. Max Planck Institute for Empirical Aesthetics
      29. Cyceron, UMS 3408 (CNRS - UCBN), France
      30. Speech & Hearing Bioscience & Technology Program, Harvard University
      31. Max Planck UCL Centre for Computational Psychiatry and Ageing Research, University College London
      32. State Key Laboratory of Cognitive Neuroscience and Learning, Beijing Normal University
      33. Department of Radiology, CHUV, Université de Lausanne

20.1.4 (July 16, 2021)
======================
Hotfix release addressing a bug on *fMRIPrep*'s resampling to standard spaces.
This release also includes some maintenance changes handling old versions of software.

* FIX: Feed *NiTransforms* with LTAs of type RAS2RAS (#2444)
* MAINT: Pin ``svgutils==0.3.1`` for the 20.1.x series (#2450)
* MAINT: Keep pip and setuptools with support for Python 3.5 (#2449)
* MAINT: Some dependencies cannot be left unpinned (#2446)

20.1.3 (September 15, 2020)
===========================
Bug-fix release in the 20.1.x series.

* FIX: Dependency conflict between *NiWorkflows* and *TemplateFlow* (#2269)
* FIX: More targetted *TemplateFlow* queries to work with all later releases (#2268)
* MAINT: Update dependency pinnings including ``niworkflows~=1.2.9`` and three minimal bug-fixes.

20.1.2 (September 04, 2020)
===========================
Bug-fix release in the 20.1.x series.

* FIX: Revise confounds in confounds-correlation plots (#2252)
* FIX: Coerce license path to pathlike (#2180)
* DOC: Update new sMRIPrep location (#2211)

20.1.1 (June 04, 2020)
======================
Bug-fix release in the 20.1.x series.

* FIX: FreeSurfer license manipulation & canary (#2165)
* FIX: Dismiss ``echo`` entity from SDC reports (#2160)
* FIX: Ensure the command-line alias of ``--nprocs`` is respected (#2152)
* MAINT: Use legacy pip/setuptools for py2 checking (#2156)

20.1.0 (May 27, 2020)
=====================
The second minor release series of 2020 is finally here!

*fMRIPrep* 20.1.0 introduces a magnitude of new features and improvements.
Originally nominated to become the first LTS (long-term support) version of *fMRIPrep*,
this release has been supercharged with many new features and bug-fixes.
To ensure long-term stability, we have postponed the LTS nomination to 20.2
to allow us unlocking the 20.1 earlier, and a more extensive stress testing of
the series before jumping into a longer support commitment.
Some key additions in this release include:

- A centralized `configuration module
  <https://fmriprep.readthedocs.io/en/latest/api.html#module-fmriprep.config>`__
  keeping track of *fMRIPrep*'s many options and run-time and environmental
  circumstances and settings.
  The new config module, which has been also propagated to other *NiPreps*
  (`dMRIPrep <https://nipreps.org/dmriprep>`__,
  `MRIQC <https://mriqc.readthedocs.io/>`__),
  comes to robustify the run-to-run replicability of *fMRIPrep* (e.g., tracking random seeds),
  make the option handling more modular but consistent (e.g., setting the ground for a
  command-line interface built off of the config module),
  and ease troubleshooting and telemetry.
- The `anatomical preprocessing fast-track
  <https://fmriprep.readthedocs.io/en/latest/usage.html#the-anatomical-fast-track>`__:
  a new experimental command-line option (``--anat-derivatives <PATH>``) checks that
  all necessary anatomical derivatives
  required by *fMRIPrep* are present under ``<PATH>``, and skips the anatomical
  processing in full if *fMRIPrep*'s expectations are met.
  Because now functional processing of many sessions and runs can be efficiently
  split into more digestible computational units (i.e., cluster job) while guaranteeing the
  exact same anatomical results are being used, this can significantly speed up
  longitudinal study preprocessing, and it is a fundamental optimization to process
  databases of densely scanned individuals such as `My Connectome
  <https://openneuro.org/datasets/ds000031>`__.
  This option is not recommended for single-session processing.
- A change in output CIFTI2 subcortical volume orientation to be compatible with HCP Pipeline tools and data.

.. admonition:: Thanks

    With thanks to Basile Pinsard, Joe B. Wexler, Noah Benson, and Marc Bue for contributions.

.. admonition:: New Paper!

    This release comes after our latest protocol paper "*Analysis of task-based
    functional MRI data preprocessed with fMRIPrep*" has been accepted.
    The protocol describes how to use *fMRIPrep* on high-performance
    clusters to preprocess fMRI data for task-based analyses.
    Please check out `the latest version on Nature Protocols
    <https://doi.org/10.1038/s41596-020-0327-3>`__ or `the preprint
    <https://doi.org/10.1101/694364>`__.

.. caution::

    As with all minor version increments, working directories
    from previous versions **should not be reused**.

Thank you for using *fMRIPrep*!
If you encounter any issues with this release, please let us know
by posting an issue on our GitHub page!

A full list of changes can be found below.

* FIX: ``MultiLabel`` interpolations should not use ``float=True`` (#2147)
* FIX: Generate proper LTA transform prior BOLD sampling on surfaces (#2146)
* FIX: Temporary config file in work directory gets clobbered in parallel jobs (#2138)
* FIX: Dismiss ``echo`` entity on several derivatives and figures outputs (#2133)
* FIX: Correct summary report when using previously run ``recon-all`` (#2124)
* FIX: Ensure correct WM and CSF masks are picked in confounds workflow (#2128)
* FIX: Explicitly add default ``native`` resolution to volumetric outputs (`nipreps/niworkflows#494`_)
* ENH: Finish the upstreaming of *NiTransforms* interfaces to *NiWorkflows* (#2132)
* ENH: Enable filtering for ``ANY`` or ``NONE`` in ``--bids-filter-file`` (#2123)
* ENH: Use new ``DerivativesDataSink`` from *NiWorkflows* 1.2.0 (#2114)
* ENH: Config module (#2018)
* ENH: Add option to ignore T2w / FLAIR images (#2015)
* ENH: Ensure subcortical volume in CIFTI is in LAS orientation (`nipreps/niworkflows#484`_)
* ENH: Add option to skip brain extraction (#2039)
* ENH: Use CIFTI sampling for carpetplot when available (#2055)
* MAINT: Stop printing full boilerplate, ``black fmriprep/cli`` (#2119)
* MAINT: Ensure YAML loader is specified (#2125)
* MAINT: PIN *tedana* version (#2117)
* MAINT: Bump minimum Python to 3.7 (#2017)
* MAINT: Remove unused console scripts (#2048)
* MAINT: Reduce the overall size of outputs (`nipreps/niworkflows#492`_)
* DOC: Update parallel subject neurostars link in FAQ (#2104)
* DOC: Add FAQ about reusing work directory (#2045)

.. _`nipreps/niworkflows#484`: https://github.com/nipreps/niworkflows/pull/484
.. _`nipreps/niworkflows#494`: https://github.com/nipreps/niworkflows/pull/494
.. _`nipreps/niworkflows#492`: https://github.com/nipreps/niworkflows/pull/492

.. admonition:: Author list for papers based on *fMRIPrep* v20.1.x series

    As described in the `Contributor Guidelines
    <https://github.com/nipreps/fmriprep/blob/e3d3bc51dbf03215e3e4d2746d8aaacdd9afb84d/CONTRIBUTING.md#publications>`__, anyone
    listed as developer or contributor may write and submit manuscripts regarding
    *fMRIPrep*.
    To do so, please move the author(s) name(s) to the front of the following list.

    Markiewicz, Christopher J. \ :sup:`1`\ ; Goncalves, Mathias \ :sup:`1`\ ; DuPre, Elizabeth \ :sup:`2`\ ; Kent, James D. \ :sup:`3`\ ; Ciric, Rastko \ :sup:`1`\ ; Salo, Taylor \ :sup:`4`\ ; de la Vega, Alejandro \ :sup:`5`\ ; Finc, Karolina \ :sup:`6`\ ; Feingold, Franklin \ :sup:`1`\ ; Tooley, Ursula A. \ :sup:`7`\ ; Benson, Noah C. \ :sup:`8`\ ; Urchs, Sebastian \ :sup:`2`\ ; Blair, Ross W. \ :sup:`1`\ ; Erramuzpe, Asier \ :sup:`9`\ ; Lurie, Daniel J. \ :sup:`10`\ ; Basile Pinsard \ :sup:`11`\ ; Heinsfeld, Anibal S. \ :sup:`12`\ ; Jacoby, Nir \ :sup:`13`\ ; Frederick, Blaise B. \ :sup:`14, 15`\ ; Valabregue, Romain \ :sup:`16`\ ; Sneve, Markus H. \ :sup:`17`\ ; Liem, Franz \ :sup:`18`\ ; Adebimpe, Azeez \ :sup:`19`\ ; Velasco, Pablo \ :sup:`20`\ ; Wexler, Joseph B. \ :sup:`1`\ ; Groen, Iris I. A. \ :sup:`21`\ ; Ma, Feilong \ :sup:`22`\ ; Rivera-Dompenciel, Adriana \ :sup:`3`\ ; Amlien, Inge K. \ :sup:`17`\ ; Cieslak, Matthew \ :sup:`19`\ ; Devenyi, Grabriel A. \ :sup:`23`\ ; Ghosh, Satrajit S. \ :sup:`24, 25`\ ; Gomez, Daniel E. P. \ :sup:`26`\ ; Halchenko, Yaroslav O. \ :sup:`22`\ ; Isik, Ayse Ilkay \ :sup:`27`\ ; Moodie, Craig A. \ :sup:`1`\ ; Naveau, Mikaël \ :sup:`28`\ ; Satterthwaite, Theodore D. \ :sup:`19`\ ; Sitek, Kevin R. \ :sup:`29`\ ; Stojić, Hrvoje \ :sup:`30`\ ; Thompson, William H. \ :sup:`1`\ ; Wright, Jessey \ :sup:`1`\ ; Ye, Zhifang \ :sup:`31`\ ; Gorgolewski, Krzysztof J. \ :sup:`1`\ ; Poldrack, Russell A. \ :sup:`1`\ ; Esteban, Oscar \ :sup:`1`\ .

    Affiliations:

      1. Department of Psychology, Stanford University
      2. Montreal Neurological Institute, McGill University
      3. Neuroscience Program, University of Iowa
      4. Department of Psychology, Florida International University
      5. University of Texas at Austin
      6. Centre for Modern Interdisciplinary Technologies, Nicolaus Copernicus University in Toruń
      7. Department of Neuroscience, University of Pennsylvania, PA, USA
      8. Department of Psychology, New York University
      9. Computational Neuroimaging Lab, BioCruces Health Research Institute
      10. Department of Psychology, Columbia University
      11. Department of Psychology, University of California, Berkeley
      12. SIMEXP Lab, CRIUGM, University of Montréal, Montréal, Canada
      13. Child Mind Institute
      14. CENIR, INSERM U1127, CNRS UMR 7225, UPMC Univ Paris 06 UMR S 1127, Institut du Cerveau et de la Moelle épinière, ICM, F-75013, Paris, France
      15. McLean Hospital Brain Imaging Center, MA, USA
      16. Consolidated Department of Psychiatry, Harvard Medical School, MA, USA
      17. Center for Lifespan Changes in Brain and Cognition, University of Oslo
      18. URPP Dynamics of Healthy Aging, University of Zurich
      19. Perelman School of Medicine, University of Pennsylvania, PA, USA
      20. Center for Brain Imaging, New York University
      21. Department of Psychology, New York University, NY, USA
      22. Dartmouth College: Hanover, NH, United States
      23. Department of Psychiatry, McGill University
      24. McGovern Institute for Brain Research, MIT, MA, USA
      25. Department of Otolaryngology, Harvard Medical School, MA, USA
      26. Donders Institute for Brain, Cognition and Behaviour, Radboud University Nijmegen
      27. Max Planck Institute for Empirical Aesthetics
      28. Cyceron, UMS 3408 (CNRS - UCBN), France
      29. Speech & Hearing Bioscience & Technology Program, Harvard University
      30. Max Planck UCL Centre for Computational Psychiatry and Ageing Research, University College London
      31. State Key Laboratory of Cognitive Neuroscience and Learning, Beijing Normal University

20.0.x series (February 2020)
=============================
20.0.7 (May 5, 2020)
--------------------
Bug-fix release in the 20.0.x series.

This release includes a new, portable version of the templateflow python client. This includes an
automatic check to fetch the latest templateflow templates every time.

* MAINT: Bump templateflow to auto-update template skeleton

20.0.6 (April 16, 2020)
-----------------------
Bug-fix release in the 20.0.x series.

This release fixes a bug for **phase-difference fieldmaps that are not in RAS+ orientation**.
The bug presented as an error if the orientation was reordered relative to RAS+ (for example,
AIL+) and the swapped dimensions were not of the same size.
Otherwise, the bug introduced a poor masking of the phase difference map, and could be quite subtle
if the original orientation was LAS+.
Runs of fMRIPrep that used other susceptibility distortion correction (SDC) methods are not
currently considered problematic.

This bug affects all earlier versions of fMRIPrep, except for 1.5.10 and any future releases in
the 1.5.x series.

  * FIX: Do not reorient magnitude images (`nipreps/sdcflows#98`_)

.. _`nipreps/sdcflows#98`: https://github.com/nipreps/sdcflows/pull/98

20.0.5 (March 19, 2020)
-----------------------
Bug-fix release in 20.0.x series.

With thanks to James Kent for the fix and Blaise Frederick for the report and testing.

  * FIX: Add CE agent to output figure filename templates (`nipreps/niworkflows#482`_)

.. _`nipreps/niworkflows#482`: https://github.com/nipreps/niworkflows/pull/482

20.0.4 (March 17, 2020)
-----------------------
A bug-fix release improving documentation for filtering BIDS files and standardizing CIFTI volume orientation.

With thanks to Ursula Tooley for the contribution.

  * DOC: FAQ section for BIDS filter (#2028)
  * FIX: Ensure BOLD and label orientations are equal (`nipreps/niworkflows#477`_).

.. _`nipreps/niworkflows#477`: https://github.com/nipreps/niworkflows/pull/477

20.0.3 (March 12, 2020)
-----------------------
A bug-fix release for CIFTI surfaces.

This release remedies a resampling error when generating fsLR surfaces that was producing erroneous CIFTI files.
**We strongly recommend all users who have generated CIFTI output with previous 20.0.x releases to upgrade and rerun**.

   * FIX: Remedy fsLR surface resampling (#2032)

20.0.2 (March 6, 2020)
----------------------
A bug squashing release in the 20.0.x series.

This release fixes the use of custom templates within the docker wrapper, remedies crashes
when FreeSurfer HOME was not set, and improves the documentation for local installations.

With thanks to Blaise Frederick for the contribution.

  * DOC: Update standalone installation requirements (#2009)
  * FIX: Crashes whenever FREESURFER_HOME is not set (#2014)
  * FIX: Local template mounting (wrapper) (#2020)
  * MAINT: Pin minor series of nipype, major series of nibabel (#2021)

20.0.1 (February 27, 2020)
--------------------------
Bug-fix release in 20.0.x series.

This release includes fixes for rare images with invalid qform matrices and some minor
improvements in report readability and inclusion of common templates in the Docker image.

  * FIX: Handle qforms with invalid quaternions (`nipreps/niworkflows#466`_)
  * FIX: update niworkflows location (#2005)
  * ENH: Display errors as summary/details elements in reports (`nipreps/niworkflows#464`_)
  * DOC: Add ``--fs-subjects-dir`` usage to slurm example (#2003)
  * CI: Test that Docker image can run a common set of output spaces without network access (#1997)

.. _`nipreps/niworkflows#464`: https://github.com/nipreps/niworkflows/pull/464
.. _`nipreps/niworkflows#466`: https://github.com/nipreps/niworkflows/pull/466

20.0.0 (February 24, 2020)
--------------------------
The major release of 2020 is here!

*fMRIPrep* is transitioning to a calendar version system
(`#1912 <https://github.com/nipreps/fmriprep/issues/1912>`__).
The `CalVer <https://calver.org/>`__ system reflects *fMRIPrep*'s nature
as an evolving workflow and does not impose any artificial incentive for
"big-change" releases.
It also permits to quickly see how out-of-date someone's version is.
As of now, the *default* version increment is the minor release number.
Hence, when the minor release number changes the work directory of *fMRIPrep*
will presumably break.
Micro releases only include bug-fixes that can reuse exiting working directories.

The major highlight of this release entails CIFTI generation to match
:abbr:`HCP (Human Connectome Project)` *grayordinates*.
In addition, the new *fMRIPrep 20.0.0* has gone through a major overhaul in the
handling of standard spaces (spatial normalizations, fusion of prior knowledge from
corresponding atlases) and imaging outputs.
In particular, the new series almost completely implements the new
syntax for ``--output-spaces`` to describe the (non)standard spatial references
that shall be used for generating outputs
(`#1604 <https://github.com/nipreps/fmriprep/issues/1604>`__).

This release includes contributions from Azeez Adebimpe and Basile Pinsard - very much appreciated.

  * ENH: Warn when existing output version does not match current pipeline version (#1967)
  * ENH: Add ``--clean-workdir`` argument (#1966)
  * ENH: Refactor of how spatial normalization targets and ``--output-spaces`` are maintained (#1955) (#1983)
  * ENH: Add ``--bids-filter-file`` argument for more controlled data querying (#1770)
  * FIX: Ensure subject ID is used when selecting BIDS data (#1982)
  * FIX: Display a log message when processing completes successfully (#1977)
  * DOC: Clean up surface outputs (#1993)
  * DOC: Integrate intersphinx, drop external module wrapping (#1989)
  * DOC: Improve custom template usage description (#1969)
  * MAINT: Use local docker registry (#1990)
  * MAINT: Pin connectome-workbench 1.3.2, add to documented dependencies (#1958)
  * MAINT: Pin NiBabel, NiWorkflows, sMRIPrep (#1971)
  * MAINT: CI build error fixes (#1976)

.. admonition:: Author list for papers based on *fMRIPrep* v20.0.0

    As described in the `Contributor Guidelines
    <https://github.com/nipreps/fmriprep/blob/d65cfdd80443c5ca779680b1087d14f189e8ceb5/CONTRIBUTING.md#publications>`__, anyone
    listed as developer or contributor may write and submit manuscripts regarding
    *fMRIPrep*.
    To do so, please move the author(s) name(s) to the front of the following list.

    Markiewicz, Christopher J.\ :sup:`1`\ ; DuPre, Elizabeth\ :sup:`2`\ ; Goncalves, Mathias\ :sup:`1`\ ; Kent, James D.\ :sup:`3`\ ; Ciric, Rastko\ :sup:`1`\ ; Salo, Taylor\ :sup:`4`\ ; de la Vega, Alejandro\ :sup:`5`\ ; Finc, Karolina\ :sup:`6`\ ; Feingold, Franklin\ :sup:`1`\ ; Urchs, Sebastian\ :sup:`2`\ ; Blair, Ross W.\ :sup:`1`\ ; Erramuzpe, Asier\ :sup:`7`\ ; Valabregue, Romain\ :sup:`8`\ ; Jacoby, Nir\ :sup:`9`\ ; Lurie, Daniel J.\ :sup:`10`\ ; Heinsfeld, Anibal S.\ :sup:`11`\ ; Halchenko, Yaroslav O.\ :sup:`12`\ ; Sneve, Markus H.\ :sup:`13`\ ; Devenyi, Grabriel A.\ :sup:`14`\ ; Liem, Franz\ :sup:`15`\ ; Gomez, Daniel E. P.\ :sup:`16`\ ; Adebimpe, Azeez\ :sup:`17`\ ; Velasco, Pablo\ :sup:`18`\ ; Groen, Iris I. A.\ :sup:`19`\ ; Ma, Feilong\ :sup:`12`\ ; Rivera-Dompenciel, Adriana\ :sup:`3`\ ; Amlien, Inge K.\ :sup:`13`\ ; Cieslak, Matthew\ :sup:`17`\ ; Ghosh, Satrajit S.\ :sup:`20, 21`\ ; Isik, Ayse Ilkay\ :sup:`22`\ ; Moodie, Craig A.\ :sup:`1`\ ; Naveau, Mikaël\ :sup:`23`\ ; Satterthwaite, Theodore D.\ :sup:`17`\ ; Sitek, Kevin R.\ :sup:`24`\ ; Stojić, Hrvoje\ :sup:`25`\ ; Thompson, William H\ :sup:`1`\ ; Tooley, Ursula A.\ :sup:`26`\ ; Wright, Jessey\ :sup:`1`\ ; Ye, Zhifang\ :sup:`27`\ ; Gorgolewski, Krzysztof J.\ :sup:`1`\ ; Poldrack, Russell A.\ :sup:`1`\ ; Esteban, Oscar\ :sup:`1`\ .

    Affiliations:

      1. Department of Psychology, Stanford University
      2. Montreal Neurological Institute, McGill University
      3. Neuroscience Program, University of Iowa
      4. Department of Psychology, Florida International University
      5. University of Texas at Austin
      6. Centre for Modern Interdisciplinary Technologies, Nicolaus Copernicus University in Toruń
      7. Computational Neuroimaging Lab, BioCruces Health Research Institute
      8. CENIR, INSERM U1127, CNRS UMR 7225, UPMC Univ Paris 06 UMR S 1127, Institut du Cerveau et de la Moelle épinière, ICM, F-75013, Paris, France
      9. Department of Psychology, Columbia University
      10. Department of Psychology, University of California, Berkeley
      11. Child Mind Institute
      12. Dartmouth College: Hanover, NH, United States
      13. Center for Lifespan Changes in Brain and Cognition, University of Oslo
      14. Department of Psychiatry, McGill University
      15. URPP Dynamics of Healthy Aging, University of Zurich
      16. Donders Institute for Brain, Cognition and Behaviour, Radboud University Nijmegen
      17. Perelman School of Medicine, University of Pennsylvania, PA, USA
      18. Center for Brain Imaging, New York University
      19. Department of Psychology, New York University, NY, USA
      20. McGovern Institute for Brain Research, MIT, MA, USA
      21. Department of Otolaryngology, Harvard Medical School, MA, USA
      22. Max Planck Institute for Empirical Aesthetics
      23. Cyceron, UMS 3408 (CNRS - UCBN), France
      24. Speech & Hearing Bioscience & Technology Program, Harvard University
      25. Max Planck UCL Centre for Computational Psychiatry and Ageing Research, University College London
      26. Department of Neuroscience, University of Pennsylvania, PA, USA
      27. State Key Laboratory of Cognitive Neuroscience and Learning, Beijing Normal University

1.5.x series (September 2019)
=============================
1.5.10 (April 16, 2020)
-----------------------
Bug-fix release in the 1.5.x series.

This release fixes a bug for **phase-difference fieldmaps that are not in RAS+ orientation**.
The bug presented as an error if the orientation was reordered relative to RAS+ (for example,
AIL+) and the swapped dimensions were not of the same size.
Otherwise, the bug introduced a poor masking of the phase difference map, and could be quite subtle
if the original orientation was LAS+.
Runs of fMRIPrep that used other susceptibility distortion correction (SDC) methods are not
currently considered problematic.

This bug affects all previous versions of fMRIPrep, as well as versions 20.0.0-20.0.5.

  * FIX: Do not reorient magnitude images (`nipreps/sdcflows#98`_)

.. _`nipreps/sdcflows#98`: https://github.com/nipreps/sdcflows/pull/98

1.5.9 (February 14, 2020)
-------------------------
Bug-fix release in the 1.5.x series.

This release fixes a bug for some phase maps generated by Philips. A full fix with better handling
of all phase maps will be available in an upcoming minor release (20.0 or 20.1), but this should permit
users who are processing with 1.5.x to resolve this issue in a way that does not affect phase maps
unaffected by the bug.

  * FIX: Center phase maps around central mode, avoiding FoV-related outliers (nipreps/sdcflows#89)

1.5.8 (January 28, 2020)
------------------------
Bug-fix release in the 1.5.x series.

  * FIX: SyN SDC logic failing in ``--force-syn`` cases (#1951)

1.5.7 (January 23, 2020)
------------------------
Bug-fix release in the 1.5.x series.

This release fixes a bug specifically for T1w images with dimensions ≤256 voxels
but a field-of-view >256mm.

  * FIX: Calculate FoV with shape and zooms (nipreps/smriprep#161)

1.5.6 (January 22, 2020)
------------------------
Bug-fix release in the 1.5.x series.

  * FIX: Include all functional runs in reports, establish consistent ordering (#1937)
  * FIX: Use SyN-SDC if --use-syn-sdc and --ignore fieldmaps are used (#1942)

1.5.5 (January 14, 2020)
-------------------------
Bug-fix release in the 1.5.x series.

* FIX: Correctly select volumetric spaces for carpetplot (#1932) @effigies
* FIX: Constrain setuptools for Python 2.7 installs of fmriprep-docker (#1933) @effigies

1.5.4 (December 18, 2019)
-------------------------
Bug-fix release in the 1.5.x series.

* FIX: Integrate fix for nipreps/sdcflows#77 (pin niworkflows-1.0.3, sdcflows-1.0.3) @oesteban

1.5.3 (December 12, 2019)
-------------------------
The last patch release of the 1.5.x series containing features.
As of 1.5.4, patch releases will only contain bug fixes, maintenance
tasks and minor documentation revisions.

* FIX: Do not run STC if SliceTiming metadata is set but empty (#1854) @oesteban
* FIX: Link to EPINorm issue in README (#1903) @adelavega
* FIX: Respect ``--dummy-scans 0`` (#1908) @jdkent
* ENH: Upgrade SDCFlows to new API (1.0.0) (#1886) @oesteban
* ENH: Add ``--fs-subjects-dir`` flag (#1901) @effigies
* DOC: Improving accessibility of confounds description (#1877) @kfinc
* MAINT: Ensure data is packaged in sdist (#1902) @effigies
* MAINT: Remove deprecated command-line arguments (#1909) @mgxd

1.5.2 (December 2, 2019)
------------------------
Bug-fix release in the 1.5.x series.

* FIX: Ensure data type of masked image matches T1.mgz (nipreps/niworkflows#430) @effigies

1.5.1 (November 26, 2019)
-------------------------
After an arduous walk through release-candidates, release 1.5.1 includes a new release of
Nipype which addresses the problems related to *results* files many users have been experiencing.

With thanks to Marc Bue, Alejandro De La Vega, Tailor Salo, Asier Erramuzpe and Soichi Hayashi.

* FIX: Treat missing field maps as empty list instead of ``None`` (#1820) @tsalo
* FIX: Raise error if ``work_dir`` is a child of ``bids_dir`` (#1860) @adelavega
* FIX: Change ICA-AROMA filenames to fit current naming scheme (#1861) @jdkent
* FIX: Update code-server in the Dockerfile_devel (#1852) @erramuzpe
* FIX: Do not generate ``desc-smoothAROMAnonaggr_bold`` conversions on standard spaces (#1838) @oesteban
* FIX: Skip plotting step of  ICA-AROMA (#1834) @oesteban
* FIX: Error during version check trying to access read-only file systems (#1830) @oesteban
* FIX: Bad results files loads; PIN: nipy/nipype master (#1806) @effigies
* FIX: Adding comma to outputnode in init_func_preproc_wf (#1795) @marcbue
* FIX: Ignore sourcedata and derivatives when fetching data (#1788) @effigies
* ENH: Added skip_citation_process flag to skip processing ``citation.md`` (#1876) @soichih
* ENH: Restore ``space-MNI152NLin6Asym`` for AROMA denoised outputs (#1839) @oesteban
* ENH: Confounds metadata (#1708) @rciric
* DOC: Remove OpenNeuro badge (#1862) @adelavega
* DOC: Improve documentation about TemplateFlow and Containers (#1802) @oesteban
* DOC: Add ``VERSION`` argument to docker build instructions (#1797) @effigies
* DOC: Revise docstrings of workflows for correct parsing with napoleon #1882 (@oesteban)
* CI: Use recent Python image to build packages (#1790) @effigies
* MAINT: Update to the new API of *sMRIPrep* (#1879) @adelavega
* MAINT: Update ``CONTRIBUTING.md`` inspired by *dMRIPrep*'s (#1853) @oesteban
* MAINT: Enable circleci-artifacts-redirector (#1857) @effigies
* MAINT: Cleaning up dependencies (#1832) @oesteban
* MAINT: Pin Python 3.7.4 in CircleCI's ``build_docs`` (#1836) @oesteban
* MAINT: Purge Cython and depend on SDCflows (#1792) @effigies
* MAINT: Container images - cleanup ``$HOME`` in docker build (#1768) @oesteban

1.5.0 (September 9, 2019)
-------------------------
Two hallmark changes conducive to a new minor release line have been included in
version 1.5.0: the upgrade of *PyBIDS* to the 0.9 series and the split of *SDCflows*
off from *fMRIPrep* codebase.
*PyBIDS* 0.9.x has a better handling of the indexed dataset that will permit some
optimizations to *fMRIPrep*'s memory fingerprint.
*SDCflows* now are found at `nipreps/sdcflows <https://github.com/nipreps/sdcflows>`__,
and have been split to allow a more granular and thorough testing in collaboration
with M. Cieslak, A. Adebimpe, and T. Satterthwaite.
Some other bugfixes, improvements to the documentation and minor features are also
shipped with the new release.
With thanks to Ursula Tooley, Sebastian Urchs and Gabriel A. Devenyi for contributions.

* FIX: Minor improvements for templateflow installation in Docker images (#1764) @oesteban
* FIX: Passing surface template keywords into carpetplot workflow (#1755) @oesteban
* FIX: Postpone ``pandoc`` conversion of boilerplate after workflow has fully run (#1710) @oesteban
* FIX: Use MNI152NLin2009cAsym for SDC if no templates are specified (#1703) @effigies
* FIX: Correct BOLD-T1w registration description if DoF != 9 (#1701) @effigies
* ENH: Set up code-server docker recipe for interactive development (#1730) @jdkent
* ENH: Make fmriprep print defaults for arguments with help (#1735) @gdevenyi
* ENH: Add check for updates and check whether version has been flagged. (#1715) @oesteban
* ENH: Add TaskName metadata to BOLD outputs (#1714) @effigies
* REF: Separate *SDCflows* from *fMRIPrep* (#1670) @oesteban
* DOC: Adding *fMRIPrep* benchmark info to FAQs (#1759) @surchs
* DOC: Update ``.zenodo.json`` (#1732) @utooley
* DOC: Add link to documentation in help of ``--output-spaces`` argument (#1722) @oesteban
* DOC: adding recon-all issue to faq (#1622) @franklin-feingold
* DOC: Add copyright waiver to boilerplate and reviewer note to docs (#1691) @effigies
* MAINT: niworkflows and nipype to use latest ``N4BiasFieldCorrection`` (#1752) @oesteban
* MAINT: Bump pybids and nipype dependencies (#1744) @effigies
* MAINT: Move regression tests of EPI masks over to Niworkflows (#1716) @oesteban
* MAINT: Remove old ``extensions`` entity selector for PyBIDS queries (#1707) @oesteban
* MAINT: Use PyBIDS 0.9.x via niworkflows/smriprep PRs (#1695) @effigies

1.4.x series (May 2019)
=======================
1.4.1 (July 9, 2019)
--------------------
As of 1.4.1, the new infant and pediatric templates added to TemplateFlow are available to
brain extraction and spatial normalization.
Containers do not set the ``TEMPLATEFLOW_HOME`` environment variable anymore, allowing
increased flexibility when running them (especially Singularity, for which the filesystem
is generally set read-only).
Additionally, BOLD files in native space can be generated and a minor bug related to the
handling of FreeSurfer outputs have been included.

* FIX: Finalizing support for new templates and their command line modifiers (#1671) @oesteban
* FIX: Do not set ``TEMPLATEFLOW_HOME`` (#1669) @oesteban
* FIX: FreeSurfer failed without adding some ``fs*`` to ``--output-spaces`` (#1643) @oesteban
* ENH: Show informative warning for phase1/2 type of fieldmaps (#1689) @oesteban
* ENH: Allow template modifiers (a la ``--output-spaces``) in skull-stripping (#1666) @oesteban
* ENH: Write outputs in native-BOLD space (#1646) @oesteban
* DOC: Add WHT to Zenodo (#1683) @wiheto

1.4.0 (May 15, 2019)
--------------------
The new 1.4 series include several new features, several maintenance patches,
and numerous bugfixes.
The largest change to *fMRIPrep*'s interface is the new ``--output-spaces``
argument that allows running spatial normalization to one or more standard
templates, and also to indicate that data preprocessed and resampled to the
individual's anatomical space should be generated.
The implementation of this option will be completed in future releases to include
new nonstandard spaces (e.g., this BOLD run's native space) and custom templates
providing a path.
For example, the following *fMRIPrep* options: ::

  --template MNI152NLin6Asym --output-space template T1w fsaverage5 ----template-resampling-grid 2mm

now would be accomplished with: ::

  --output-spaces MNI152NLin6Asym:res-2 anat fsaverage:den-10k

with the difference that more templates could be specified if needed, e.g., ::

  --output-spaces MNI152NLin6Asym:res-2 anat fsaverage:den-10k MNI152NLin2009cAsym:native

Related anatomical preprocessing workflows from *sMRIPrep* have gone through
thorough revisions.
In particular, the brain extraction workflow now is implemented in pure Nipype.

Users will notice the addition of two new subsections in the reports generated by
*fMRIPrep*.
The first addition describes the cumulative variance explained by successive a/tCompCor
components.
A second addition shows the correlations between the confounding regressors that
*fMRIPrep* writes to the corresponding file, and their correlation to the global signal.

Series 1.4 increasingly relies on PyBIDS to handle not only inputs, but also outputs and
reporting.
The reports generation system has been deeply refactored to improve its generalizability
across BIDS-Apps and addressing some rendering problems (e.g., when resizing ICA-AROMA
components decompositions).
Finally, there were several updates to packaging, testing and documentation, which should
hopefully improve the experience for new users and contributors.

With thanks to Yaroslav Halchenko, Dan Lurie, Adriana Rivera-Dompenciel, Franklin Feingold,
Markus Sneve, Anibal Heinsfeld, and James Kent for contributions.

* FIX: Incorrect transforms being applied to aparc/asegs in standard space (#1636) @oesteban
* FIX: Writing functional derivatives in standard spaces (#1632) @oesteban
* FIX: Resampling of BOLD into standard spaces (#1627) @oesteban
* FIX: Re-enable correct versioning within container (#1623) @oesteban
* FIX: Update spline fitting distance for BOLD bias-field correction (#1603) @markushs
* FIX: Mount Nipype config file under new ``$HOME`` (#1602) @oesteban
* FIX: Not having ``template`` as one ``--output-space`` crashes fMRIPrep (#1560) @oesteban
* ENH: Add ``--dummy-scans`` option (#1559) @jdkent
* ENH: Reduce ``BSplineFieldmap`` memory usage (#1609) @effigies
* ENH: Confound model enhancement (#1487, #1586) @rciric
* ENH: Allow multiple {non,}standard spaces (``--output-spaces``) (#1596) @oesteban
* ENH: Switch to the refactored report generation from NiWorkflows (#1599) @oesteban
* ENH: Force compression of derivative NIfTI volumes (#1600) @effigies
* ENH: Do not allow writing derivatives directly into the BIDS root folder (#1589) @oesteban
* ENH: Support 4D SBRefs when generating the bold reference (#1581) @oesteban
* ENH: Remove pre-existing citation files before running (#1567) @yarikoptic
* DOC: Improve appearance of parameter types in API docs (#1633) @anibalsolon
* DOC: Confound enhancement documentation (#1625) @rciric
* DOC: Add FAQ, Tips, Tricks section to RTD (cont. #1601) (#1610) @franklin-feingold
* DOC: Companion of #1596 + punctual improvements of docs (#1605) @oesteban
* DOC: Add examples/clarify ``CONTRIBUTORS.md`` (#1566) @jdkent
* DOC: Add ``.bidsignore`` requirement to docs on lesion masks (#1574) @danlurie
* DOC: Bump references of python3.6 to python3.7 (#1562) @jdkent
* MAINT: Consolidate build configuration in ``setup.cfg`` (#1607) @effigies
* MAINT: Progress bids-validator 1.2.3 (#1583) @yarikoptic
* MAINT: Pin ``nilearn!=0.5.0,!=0.5.1`` to avoid problems plotting mgz images (#1585) @oesteban
* MAINT: Group all 'finished running with errors' sentry messages (#1552) @chrisgorgo
* MAINT: Include hash of ``dataset_description.json`` in sentry metadata (#1553) @chrisgorgo
* CI: Reduce pointless expenditures (#1614) @effigies
* CI: Use caches to pass data between jobs (#1608) @effigies
* CI: Build docs outside of container (#1606) @effigies
* CI: Fix label on CI step "Skipping doc building job" (#1564) @ariveradompenciel

1.3.x series (March 2019)
=========================
1.3.2 (March 18, 2019)
----------------------
A new release providing better support for BIDS-Derivatives, an wrapping-up all the developments around TemplateFlow and stability of singularity images. With thanks to @sarenseeley for contributions.

* ENH: Write derivatives metadata (#1546) @oesteban
* DOC: Indicate that interpolation is NN above the EPI-to-T1w reportlet (#1542) @sarenseeley

1.3.1.post2 (March 13, 2019)
----------------------------
A hotfix release addressing issues related to TemplateFlow for Singularity users, via pinning templateflow>=0.1.2.

1.3.1.post1 (March 11, 2019)
----------------------------

A hotfix release addressing issues related to TemplateFlow for Singularity users.

* FIX: Make sure ``--cifti-output`` requires at least one of ``fsaverage{5,6}`` (#1514) @oesteban
* FIX: Avoid using ``$HOME`` for storing templates (#1529) @chrisfilo

1.3.1 (March 6, 2019)
---------------------
Updated ecosystem's versions (TemplateFlow 0.1.x, Niworkflows 0.8.x, and sMRIPrep 0.1.x)
to include latest improvements (bugfixes from niworkflows and the new pybids interface
of templateflow).
TemplateFlow 0.1.0 does not require datalad anymore.
With thanks to @franklin-feingold for contributions.

* UX: Reduce warning levels (#1513) @effigies
* DOC: ``fmriprep-docker`` documentation (#1515) @franklin-feingold
* REL: 1.3.1 (#1527) @oesteban

1.3.0.post3 (March 1, 2019)
---------------------------
Hotfix release intended for Docker users, smoothing the experience of TemplateFlow
when using the ``-u UID`` flag is necessary.

* FIX: Orientation problem with niworkflows<0.7.2 (nipreps/niworkflows#312) @effigies
* ENH: TemplateFlow + ``docker run -u ...`` (#1525) @oesteban
* ENH: Include repetition time in functional summary (#1508) @wiheto
* FIX: Do not crash if all aroma components are classified as noise (#1467) @jdkent

1.3.0.post2 (February 14, 2019)
-------------------------------
Hotfix release intended for Singularity users, amending the previous iteration
that didn't fix the problem (#1510) @effigies.

1.3.0.post1 (February 8, 2019)
------------------------------
Hotfix release intended for Singularity users. For further detail, please see
`#1500 <https://github.com/nipreps/fmriprep/issues/1500>`__.

1.3.0 (February 7, 2019)
------------------------
We start the 1.3.x series including a few bugfixes, housekeeping duty and a refactors
to leverage `sMRIPrep <https://github.com/nipreps/smriprep>`__ (which is a fork of
fMRIPrep's anatomical workflow), pybids>=0.7 for querying dataset, and
`TemplateFlow <https://github.com/templateflow>`__ for handling standard spaces.

* FIX: Bad ``fsnative`` replacement in CIfTI workflow (#1476) @oesteban
* FIX: Avoid warning when generating boilerplate (#1464) @oesteban
* MAINT: resolves #1485 : patch fmriprep-docker automount for use with Python 3.7 (#1486) @rciric
* RF: Use anatomical workflows from sMRIPrep (#1482) @oesteban
* MAINT: Update sentry-sdk (#1490) @chrisfilo
* ENH: Remaining TemplateFlow integrations (#1494) @oesteban
* MAINT: Update to keep up with nipreps/niworkflows#299 (#1496) @oesteban
* FIX: Updating bids-validator to 1.1.3 (#1498) @chrisfilo

1.2.x series (January 2019)
===========================
1.2.6-1 (January 24, 2019)
--------------------------
Hotfix release of version 1.2.6, pinning niworkflows to a release version (instead
of the development branch, since #1459) and including to bugfixes.

* PIN: NiWorkflows 0.5.2.post7 (`1bf4a21 <https://github.com/nipreps/fmriprep/commit/1bf4a21cce62c4330510a9a8ae50db876fbc23b0>`__).
* FIX: Bad ``fsnative`` replacement in CIfTI workflow (#1476) @oesteban
* FIX: Avoid warning when generating boilerplate (#1464) @oesteban

1.2.6 (January 17, 2019)
------------------------
This is a bug fix release in the 1.2 series. Probably the most noticeable
improvement is the restoration of auto-generated content in the documentation.

Additionally, FreeSurfer ``aparc``/``aseg`` segmentations are now sampled to all
output spaces.

For any users importing fMRIPrep interfaces, many of these have been moved to
the niworkflows package.

With thanks to Nir Jacoby and Hrvoje Stojic for contributions.

* FIX: Use keyword arguments for Sentry breadcrumb reporting (#1441) @chrisfilo
* FIX: Verify proc file exists before reading (#1454) @effigies
* ENH: Only report participants with errors (#1437) @effigies
* ENH: Resample aparc/aseg into specified output spaces (#1401) @nirjacoby
* ENH: Copy BibTeX file to log directory for LaTeX users (#1446) @hstojic
* RF: Use niworkflows upstreamed interfaces and utilities (#1438) @oesteban
* DOC: Fix documentation build (#1451) @oesteban
* DOC: Fix ReadTheDocs builds (#1459) @effigies
* MAINT/DOC: Clean-up ``__about__``, update with Nat Meth (#1445) @oesteban
* MAINT: Make sure Python 3.7.1 is installed (#1452) @oesteban
* MAINT: Dev status to beta, bump copyright year (#1468) @effigies

1.2.5 (December 4, 2018)
------------------------
Hotfix release.

* FIX: Breadcrumb reporting (#1435) @chrisfilo

1.2.4 (December 3, 2018)
------------------------
Bugfixes, an additional iteration over Sentry reporting and some relevant ME-EPI updates
(with thanks to @emdupre).

* ENH: Update ME-EPI workflow to create optimal combination (#1263) @emdupre
* MAINT: Merge master into multiecho (#1324) @effigies
* ENH: Add echo-idx flag (#1355) @emdupre
* FIX: Always run FreeSurfer interfaces that sink outside working directory (#1397) @effigies
* ENH: Use Python 3.7 in Dockerfile (#1398) @effigies
* DOC: Update contributing guide and add code of conduct (#1404) @emdupre
* FIX: Calculate template transforms explicitly as RAS2RAS (#1399) @effigies
* MAINT: Replace ``img.get_affine()`` -> ``img.affine`` (#1414) @oesteban
* FIX: Truncating of sentry messages (#1417) @chrisfilo
* ENH: Add fmriprep-docker execution environment (#1416) @chrisfilo
* MAINT: Update indexed_gzip to handle small .nii.gz (#1421) @effigies
* ENH: Group common issues with fingerprints (#1418) @chrisfilo
* ENH: adding memory and cpu info to sentry logs (#1420) @chrisfilo
* ENH: Use standard T2* map as coregistration target (#1383) @emdupre
* ENH: Handle FreeSurfer subject directory preparation gracefully when run in parallel (#1413) @effigies
* ENH: Make sure inputs are BIDS compliant before running fmriprep (#1419) @chrisfilo
* ENH: Sentry event categorization propagation (#1422) @chrisfilo
* MAINT: Require nipype >= 1.1.6 (#1426) @effigies
* ENH: Omnibus multi-echo pull request (#1296) @effigies
* ENH: Report memory overcommit policies (#1429) @effigies

1.2.3 (November 16, 2018)
-------------------------
Refactor of Sentry reporting, bug fixes and added tests. With thanks to @sebnaze for contributions.

* TST: Utility functions for skipping/re-inserting non-steady-state volumes (#1382) @jdkent
* FIX: Correctly populate right-hemisphere time series in CIFTI derivatives (#1378) @sebnaze
* FIX: Restore original contour colors in reports (#1385) @oesteban
* ENH: New sentry SDK (#1381) @chrisfilo
* ENH: Sentry refinement (#1394) @chrisfilo

1.2.2 (November 9, 2018)
------------------------
Several bug fixes. With thanks to Franz Liem, Nir Jacoby and Markus Handal Sneve for contributions.

* FIX: Do not show --debug deprecation warning unless used (#1361) @effigies
* FIX: Select consistent parcellation for producing aparcaseg derivatives (#1369) @nirjacoby
* FIX: Count non-steady-state volumes even if sbref is passed (#1373) @effigies
* ENH: Respect SliceEncodingDirection metadata (#1350) @fliem
* ENH: Set maximum MELODIC components to 200 by default (#1366) @markushs
* TST: Verify LegacyMultiProc functionality (#1368) @effigies

1.2.1 (November 1, 2018)
------------------------
Hotfix release (deployment system)

1.2.0 (October 31, 2018)
------------------------
This release marks a substantial renaming of derivatives to conform to the BIDS Derivatives specification [release candidate](https://docs.google.com/document/d/17ebopupQxuRwp7U7TFvS6BH03ALJOgGHufxK8ToAvyI/).

The most significant additional change is a substantial revision of BOLD skull-stripping, using a BOLD template constructed from many open datasets. Building off the work of Zhifang Ye (see #1050), the skull-stripping is now much more resilient to intensity inhomogeneity.

With many thanks to Ali Cohen, James Kent, Inge Amlien, Sebastian Urchs, and Zhifang Ye for contributions.

* FIX: Missing BOLD reports (#1326) @oesteban
* FIX: Ensure encoding when reading boilerplate (#1322) @alioco
* FIX: Reportlets - bbregister vs flirtbbr (continues #1326) (#1328) @oesteban
* FIX: Quick update to new template structure (#1330) @oesteban
* FIX: Explicitly pass bold mask to AROMA (#1332) @jdkent
* FIX: Missing report output - #1339 (#1346) @kasbohm
* FIX: Remove non-steady-state volumes prior to ICA-AROMA (#1335) @jdkent
* ENH: Store BOLD reference images (#1306) @oesteban
* ENH: Deprecate --debug with --sloppy (#1347) @effigies
* ENH: Conform confound regressor names to Derivatives RC2 (#1343) @effigies
* ENH: Do not set KEEP_FILE_OPEN_DEFAULT (#1356) @effigies
* ENH: Template-based masking of EPI boldrefs (#1321) @oesteban
* DOC: Update BIDS-validator link (#1320) @surchs
* DOC: add --bind method to singularity patch documentation (#1340) @jdkent
* RF: Update anatomical derivatives for RC1  (#1325) @effigies
* RF: Update functional derivatives for RC1 (#1333) @effigies
* TST: Add heavily-nonuniform boldrefs for regression tests (#1329) @oesteban
* TST: Fix expectations for CIFTI outputs & ds005 (#1344) @oesteban
* MAINT: Ignore project settings files from popular python/code editors (#1336) @jdkent
* CI: Deploy nipreps/fmriprep:unstable tracking master (#1307) @effigies

1.1.x series (October 2018)
===========================
1.1.8 (October 4, 2018)
-----------------------
Several bug fixes. This release is intended to be the last before start
adopting BIDS-Derivatives RC1 (which will trigger 1.2.x versions).

* DOC: Switch to orig graph for ``init_bold_t2s_wf`` (#1298) @effigies
* FIX: Enhance T2 contrast ``enhance_t2`` in reference estimate (#1299) @effigies
* FIX: Create template from one usable T1w image (#1305) @effigies
* MAINT: Pin grabbit and pybids in ``setup.py`` (#1284) @oesteban

1.1.7 (September 25, 2018)
--------------------------
Several bug fixes. With thanks to Elizabeth Dupre and Romain Vala for
contributions.

* FIX: Revert FreeSurfer download URL (#1280) @chrisfilo
* FIX: Default to 6 DoF for BOLD-T1w registration (#1286) @effigies
* FIX: Only grab sbref images, not metadata (#1285) @effigies
* FIX: QwarpPlusMinus renamed source_file to in_file (#1289) @effigies
* FIX: Remove long paths from all LTA output files (#1274) @romainVala
* ENH: Use single-band reference images when available (#1270) @effigies
* DOC: Note GIFTI surface alignment (#1288) @effigies
* RF: Split BOLD-T1w registration into calculation/application workflows (#1278) @emdupre
* MAINT: Pin pybids and grabbit in Docker build (#1281) @chrisfilo

1.1.6 (September 10, 2018)
--------------------------
Hotfix release.

* FIX: Typo in plugin config loading.

1.1.5 (September 06, 2018)
--------------------------
Improved documentation and minor bug fixes. With thanks to Jarod Roland and
Taylor Salo for contributions.

* DOC: Replace ``--clearenv`` with correct ``--cleanenv`` flag (#1237) @jarodroland
* DOC: De-indent to remove text from code block (#1238) @effigies
* TST: Add enhance-and-skullstrip regression tests (#1074) @effigies
* DOC: Clearly indicate that fMRIPrep requires Python 3.5+ (#1249) @oesteban
* MAINT: Update PR template (#1239) @effigies
* DOC: Set appropriate version in Zenodo citation (#1250) @oesteban
* DOC: Updating long description (#1230) @oesteban
* DOC: Add ME workflow description (#1253) @tsalo
* FIX: Add memory annotation to ROIPlot interface (#1256) @jdkent
* ENH: Write derivatives ``dataset_description.json`` (#1247) @effigies
* DOC: Enable table text wrap and link docstrings to code on GitHub (#1258) @tsalo
* DOC: Clarify language describing T1w image merging (#1269) @chrisfilo
* FIX: Accommodate new template formats (#1273) @effigies
* FIX: Permit overriding plugin config with CLI options (#1272) @effigies


1.1.4 (August 06, 2018)
-----------------------
A hotfix release for `#1235
<https://github.com/nipreps/fmriprep/issues/1235>`_. Additionally,
notebooks have been synced with the latest version of that repository.

* FIX: Verify first word of ``_cmd`` in dependency check (#1236)
* DOC: Add two missing references (#1234)
* ENH: Allow turning off random seeding for ANTs brain extraction (#919)

1.1.3 (July 30, 2018)
---------------------
This release comes with many updates to the documentation, a more lightweight
``SignalExtraction``, a new dynamic boilerplate and some new features from
Nipype.

* ENH: Use upstream ``afni.TShift`` improvements (#1160)
* PIN: Nipype 1.1.1 (65078c9)
* ENH: Dynamic citation boilerplate (#1024)
* ENH: Check Command Line dependencies before running (#1044)
* ENH: Reimplement ``SignalExtraction`` (#1170)
* DOC: Update copyright year to 2018 (#1224)
* ENH: Enable ``-u`` (docker user/userid) flag in wrapper (#1223)
* FIX: Corrects Dockerfile ``WORKDIR``. (#1218)
* ENH: More specific errors for missing echo times (#1221)
* ENH: Change ``WORKDIR`` of Docker image (#1204)
* DOC: Update documentation related to contributions (#1187)
* DOC: Additions to include before responding to reviews of the pre-print (#1195)
* DOC: Improving documentation on using Singularity (#1063)
* DOC: Add OHBM 2018 poster, presentation (#1198)
* ENH: Replace ``InvertT1w`` with upstream ``Rescale(invert=True)`` (#1161)

1.1.2 (July 6, 2018)
--------------------
This release incorporates Nipype improvements that should reduce the
chance of hanging if tasks are killed for excessive resource consumption.

Thanks to Elizabeth DuPre for documentation updates.

* DOC: Clarify how to reuse FreeSurfer derivatives (#1189)
* DOC: Improve command line option documentation (#1186, #1080)
* MAINT: Update core dependencies (#1179, #1180)

1.1.1 (June 7, 2018)
--------------------
* ENH: Pre-cache DKT31 template in Docker image (#1159)
* MAINT: Update core dependencies (#1163)

1.1.0 (June 4, 2018)
--------------------
* ENH: Use Reorient interface included upstream in nipype (#1153)
* FIX: Refine BIDS queries to avoid indexing derivatives (#1141)
* DOC: Clarify outlier columns (#1138)
* PIN: Update to niworkflows 0.4.0 and nipype 1.0.4 (#1133)

1.0.x series (May 2018)
=======================
1.0.15 (May 17, 2018)
---------------------
* DOC: Add lesion masking during registration (#1113)
* FIX: Patch ``boldbuffer`` for ME (#1134)

1.0.14 (May 15, 2018)
---------------------
With thanks to @ZhifangYe for contributions

* FIX: Non-invertible transforms bringing parcellation to BOLD (#1130)
* FIX: Bad connection for ``--medial-surface-nan`` option (#1128)

1.0.13 (May 11, 2018)
---------------------
With thanks to @danlurie for the outstanding contribution of #1106

* ENH: Some nit picks on reports (#1123)
* ENH: Carpetplot + confounds plot (#1114)
* ENH: Add constrained cost-function masking to T1-MNI registration (#1106)
* FIX: Circular dependency (#1104)
* ENH: Set ``PYTHONNOUSERSITE`` in containers (#1103)

1.0.12 (May 03, 2018)
---------------------
* MAINT: fmriprep-docker: Ensure data/output/work paths are absolute (#1089)
* ENH: Add usage tracking and centralized error reporting (#1088)
* FIX: Ensure one motion IC index is loaded as list (#1096)
* TST: Refactoring CircleCI setup (#1098)
* FIX: Compression in DataSinks (#1095)
* MAINT: fmriprep-docker: Support Python 2/3 without future or other helpers (#1082)
* MAINT: Update npm to 10.x (#1087)
* DOC: Prefer pre-print over Zenodo doi in boilerplate (#1086)
* DOC: Stylistic fix (\`'template'\`) (#1083)
* FIX: Run ICA-AROMA in ``MNI152Lin`` 2mm resampling grid (91x109x91 vox) (#1064)
* MAINT: Remove cwebp to revert to png (#1081)
* ENH: Allow changing the dimensionality of Melodic for AROMA. (#1052)
* FIX: Derivatives datasink handling of compression (#1077)
* FIX: Check for invalid sform matrices (#1072)
* FIX: Check exit code from subprocess (#1073)
* DOC: Add preprint fig. 1 to About (#1070)
* FIX: Always strip session from T1w for derivative naming (#1071)
* DOC: Add RRIDs in the citation boilerplate (#1061)
* ENH: Generate CIFTI derivatives (#1001)

1.0.11 (April 16, 2018)
-----------------------
* FIX: Do not detrend CSF/WhiteMatter/GlobalSignal (#1058)

1.0.10 (April 16, 2018)
-----------------------
* TST: Re-run ds005 with only one BOLD run (#1048)
* FIX: Patch subject_summary in reports (#1047)

1.0.9 (April 10, 2018)
----------------------
With thanks to @danlurie for contributions.

* FIX: Connect inputnode to SDC for pepolar images (#1046)
* FIX: Pass ``ref_file`` to STC check (#1038)
* DOC: Add BBR fallback to user docs. (#1036)
* ENH: Revise resampling grid for template outputs (#1040)
* MAINT: DataSinks within their workflows (#1021)
* ENH: Add FLAIR pial refinement support (#829)
* MAINT: Upgrade to pybids 0.5 (#1027)
* MAINT: Refactor fieldmap heuristics (#1017)
* FIX: Use metadata to select shortest echo as ref_file (#1018)
* ENH: Adopt versioneer to compose version names (#1007)
* ENH: Handle first echo separately for ME-EPI (#891)

1.0.8 (February 22, 2018)
-------------------------
With thanks to @mgxd and @naveau for contributions.

* FIX: ROIs Plot and output brain masks consistency (#1002)
* FIX: Init flirt with qform (#1003)
* DOC: Prepopulate tag when posting neurostars questions. (#987)
* FIX: Update fmap.py : import _get_pe_index in get_ees (#984)
* FIX: Argparse action (#985)

1.0.7 (February 13, 2018)
-------------------------
* ENH: Output ``aseg`` and ``aparc`` in T1w and BOLD spaces (#957)
* FIX: Write latest BOLD mask out (space-T1w) (#978)
* PIN: Updating niworkflows to 0.3.1 (#962)
* FIX: Robuster BOLD mask (#966)

1.0.6 (29th of January 2018)
----------------------------
* FIX: Bad connection in phasediff-fieldmap workflow (#950)
* PIN: niworkflows-0.3.1-dev (including Nipype 1.0.0!)
* ENH: Migrate to CircleCI 2.0 and workflows (#943)
* ENH: Improvements to CLIs (native & wrapper) (#944)
* FIX: Rerun tCompCor interface in case of MemoryError (#942)

1.0.5 (21st of January 2018)
----------------------------
* PIN: niworkflows-0.2.8 to fix several execution issues.
* ENH: Code cleanup (#938)

1.0.4 (15th of January 2018)
----------------------------
* FIX: Pin niworkflows-0.2.6 to fix several MultiProc errors (nipy/nipype#2368)
* DOC: Fix DOI in citation boilerplate (#933)
* FIX: Heuristics to prevent memory errors during aCompCor (#930).
* FIX: RuntimeWarning: divide by zero encountered in float_scalars (#931).
* FIX: INU correction before merging several T1w (#925).

1.0.3 (3rd of January 2018)
---------------------------
* FIX: Pin niworkflows-0.2.4 to fix (#868).
* FIX: Roll back run/task groupings after BIDS query (#918).
  Groupings for the multi-echo extension will be reenabled soon.

1.0.2 (2nd of January 2018)
---------------------------
* FIX: Grouping runs broke FMRIPREP on some datasets (#916)
  Thanks to @emdupre

1.0.1 (1st of January 2018)
---------------------------
With thanks to @emdupre for contributions.

* PIN: Update required niworkflows version to 0.2.3
* FIX: Refine ``antsBrainExtraction`` if ``recon-all`` is run (#912)
  With thanks to Arno Klein for his [helpful comments
  here](https://github.com/nipreps/fmriprep/issues/431#issuecomment-299583391)
* FIX: Use thinner contours in reportlets (#910)
* FIX: Robuster EPI mask (#911)
* FIX: Set workflow return value before potential error (#887)
* DOC: Documentation about FreeSurfer and ``--fs-no-reconall`` (#894)
* DOC: Fix example in installation ants-nthreads -> omp-nthreads (#885)
  With thanks to @mvdoc.
* ENH: Allow for multiecho data (#875)

1.0.0 (6th of December 2017)
----------------------------
* ENH: Add ``--resource-monitor`` flag (#883)
* FIX: Collision between Multi-T1w and ``--no-freesurfer`` (#880)
* FIX: Setting ``use_compression`` on resampling workflows (#882)
* ENH: Estimate motion parameters before STC (#876)
* ENH: Add ``--stop-on-first-crash`` option (#865)
* FIX: Correctly handling xforms (#874)
* FIX: Combined ROI reportlets (#872)
* ENH: Strip reportlets out of full report (#867)

1.0.0-rc13 (1st of December 2017)
---------------------------------
* FIX: Broken ``--fs-license-file`` argument (#869)

1.0.0-rc12 (29th of November 2017)
----------------------------------
* ENH: Use Nipype MultiProc even for sequential execution (#856)
* RF: More memory annotations and considerations (#816)
* FIX: Controlling memory explosion (#854)
* FIX: Mount nipype repositories as niworkflows submodule (#834)
* FIX: Reduce image loads in local memory (#839)
* ENH: Always sync qforms, refactor error messaging (#851)

1.0.0-rc11 (24th of November 2017)
----------------------------------
* ENH: Check for invalid qforms in validation (#847)
* FIX: Update pybids to include latest bugfixes (#838)
* FIX: MultiApplyTransforms failed with nthreads=1 (#835)

1.0.0-rc10 (9th of November 2017)
---------------------------------
* FIX: Adopt new FreeSurfer (v6.0.1) license mechanism (#787)
* ENH: Output affine transforms from original T1w images to preprocessed anatomical (#726)
* FIX: Correct headers in AFNI-generated NIfTI files (#818)
* FIX: Normalize T1w image qform/sform matrices (#820)

1.0.0-rc9 (2nd of November 2017)
--------------------------------
* FIX: Fixed #776 (aCompCor - numpy.linalg.linalg.LinAlgError: SVD did not converge) via #807.
* ENH: Added ``CSF`` column to ``_confounds.tsv`` (included in #807)
* DOC: Add more details on the outputs of FMRIPREP and minor fixes (#811)
* ENH: Processing confounds in BOLD space (#807)
* ENH: Updated niworkflows and nipype, including the new feature to close all file descriptors (#810)
* RF: Refactored BOLD workflows module (#805)
* ENH: Improved memory annotations (#803, #807)

1.0.0-rc8 (27th of October 2017)
--------------------------------
* FIX: Allow missing magnitude2 in phasediff-type fieldmaps (#802)
* FIX: Lower tolerance deciding t1_merge shapes (#798)
* FIX: Be robust to 4D T1w images (#797)
* ENH: Resource annotations (#746)
* ENH: Use indexed_gzip with nibabel (#788)
* FIX: Reduce FoV of outputs in T1w space (#785)

1.0.0-rc7 (20th of October 2017)
--------------------------------
* ENH: Update pinned version of nipype to latest master
* ENH: Added rX permissions to make life easier on Singularity users (#757)
* DOC: Citation boilerplate (#779)
* FIX: Patch to remove long filenames after mri_concatenate_lta (#778)
* FIX: Only use unbiased template with ``--longitudinal`` (#771)
* FIX: Use t1_2_fsnative registration when sampling to surface (#762)
* ENH: Remove ``--skull_strip_ants`` option (#761)
* DOC: Add reference to beginners guide (#763)


1.0.0-rc6 (11th of October 2017)
--------------------------------
* ENH: Add inverse normalization transform (MNI -> T1w) to derivatives (#754)
* ENH: Fall back to initial registration if BBR fails (#694)
* FIX: Header and affine transform updates to resolve intermittent
  misalignments in reports (#743)
* FIX: Register FreeSurfer template to FMRIPREP template, handling pre-run
  FreeSurfer subjects more robustly, saving affine to derivatives (#733)
* ENH: Add OpenFMRI participant sampler command-line tool (#704)
* ENH: For SyN-SDC, assume phase-encoding direction of A-P unless specified
  L-R (#740, #744)
* ENH: Permit skull-stripping with NKI ANTs template (#729)
* ENH: Erode aCompCor masks to target volume proportions, instead of fixed
  distances (#731, #732)
* DOC: Documentation updates (#748)

1.0.0-rc5 (25th of September 2017)
----------------------------------
* FIX: Skip slice time correction on BOLD series < 5 volumes (#711)
* FIX: Skip AFNI check for new versions (#723)
* DOC: Documentation clarification and updates (#698, #711)

1.0.0-rc4 (12th of September 2017)
----------------------------------
With thanks to Mathias Goncalves for contributions.

* ENH: Collapse ITK transforms of head-motion correction in only one file (#695)
* FIX: Raise error when run.py is called directly (#692)
* FIX: Parse crash files when they are stored as text (#690)
* ENH: Replace medial wall values with NaNs (#687)

1.0.0-rc3 (28th of August 2017)
-------------------------------
With thanks to Anibal Sólon for contributions.

* ENH: Add ``--low-mem`` option to reduce memory usage for large BOLD series (#663)
* ENH: Parallelize anatomical conformation step (#666)
* FIX: Handle missing functional data in SubjectSummary node (#670)
* FIX: Disable ``--no-skull-strip-ants`` (AFNI skull-stripping) (#674)
* FIX: Initialize SyN SDC more robustly (#680)
* DOC: Add comprehensive documentation of workflow API (#638)

1.0.0-rc2 (12th of August 2017)
-------------------------------
* ENH: Increased support for partial field-of-view BOLD datasets (#659)
* FIX: Slice time correction is now being applied to output data (not only to intermediate file used for motion estimation - #662)
* FIX: Fieldmap unwarping is now being applied to MNI space outputs (not only to T1w space outputs - #662)

1.0.0-rc1 (8th of August 2017)
------------------------------
* ENH: Include ICA-AROMA confounds in report (#646)
* ENH: Save non-aggressively denoised BOLD series (#648)
* ENH: Improved logging messages (#621)
* ENH: Improved resource management (#622, #629, #640, #641)
* ENH: Improved confound header names (#634)
* FIX: Ensure multi-T1w image datasets have RAS-oriented template (#637)
* FIX: More informative errors for conflicting options (#632)
* DOC: Improved report summaries (#647)

0.x series (July 2017)
======================
0.6.0 (31st of July 2017)
-------------------------
With thanks to Yaroslav Halchenko and Ilkay Isik for contributions.

* ENH: Set threshold on up-sampling ratio in conformation, report results (#601)
* ENH: Censor non-steady-state volumes prior to CompCor (#603)
* FIX: Conformation failure in thick-slice, oblique T1w datasets (#601)
* FIX: Crash/report failure of phase-difference SDC pipeline (#602, #604)
* FIX: Prevent AFNI NIfTI extensions from crashing reference EPI estimation (#619)
* DOC: Save logs to output directory (#605)
* ENH: Upgrade to ICA-AROMA 0.4.1-beta (#611)

0.5.4 (20th of July 2017)
-------------------------
* DOC: Improved report summaries describing steps taken (#584)
* ENH: Uniformize command-line argument style (#592)

0.5.3 (18th of July 2017)
-------------------------
With thanks to Yaroslav Halchenko for contributions.

* ENH: High-pass filter time series prior to CompCor (#577)
* ENH: Validate and minimally conform BOLD images (#581)
* FIX: Bug that prevented PE direction estimation (#586)
* DOC: Log version/time in report (#587)

0.5.2 (30th of June 2017)
-------------------------
With thanks to James Kent for contributions.

* ENH: Calculate noise components in functional data with ICA-AROMA (#539)
* FIX: Remove unused parameters from function node, resolving crash (#576)

0.5.1 (24th of June 2017)
-------------------------
* FIX: Invalid parameter in ``bbreg_wf`` (#572)

0.5.0 (21st of June 2017)
-------------------------
With thanks to James Kent for contributions.

* ENH: EXPERIMENTAL: Fieldmap-less susceptibility correction with ``--use-syn-sdc`` option (#544)
* FIX: Reduce interpolation artifacts in ConformSeries (#564)
* FIX: Improve consistency of handling of fieldmaps (#565)
* FIX: Apply T2w pial surface refinement at correct stage of FreeSurfer pipeline (#568)
* ENH: Add ``--anat-only`` workflow option (#560)
* FIX: Output all tissue class/probability maps (#569)
* ENH: Upgrade to ANTs 2.2.0 (#561)

0.4.6 (14th of June 2017)
-------------------------
* ENH: Conform and minimally resample multiple T1w images (#545)
* FIX: Return non-zero exit code on all errors (#554)
* ENH: Improve error reporting for missing subjects (#558)

0.4.5 (12th of June 2017)
-------------------------
With thanks to Marcel Falkiewicz for contributions.

* FIX: Correctly display help in ``fmriprep-docker`` (#533)
* FIX: Avoid invalid symlinks when running FreeSurfer (#536)
* ENH: Improve dependency management for users unable to use Docker/Singularity containers (#549)
* FIX: Return correct exit code when a Function node fails (#554)

0.4.4 (20th of May 2017)
------------------------
With thanks to Feilong Ma for contributions.

* ENH: Option to provide a custom reference grid image (``--output-grid-reference``) for determining the field of view and resolution of output images (#480)
* ENH: Improved EPI skull stripping and tissue contrast enhancements (#519)
* ENH: Improve resource use estimates in FreeSurfer workflow (#506)
* ENH: Moved missing values in the DVARS* and FramewiseDisplacement columns of the _confounds.tsv from last row to the first row (#523)
* ENH: More robust initialization of the normalization procedure (#529)

0.4.3 (10th of May 2017)
------------------------
* ENH: ``--output-space template`` targets template specified by ``--template`` flag (``MNI152NLin2009cAsym`` supported) (#498)
* FIX: Fix a bug causing small numerical discrepancies in input data voxel size to lead to different FOV of the output files (#513)

0.4.2 (3rd of May 2017)
-----------------------
* ENH: Use robust template generation for multiple T1w images (#481)
* ENH: Anatomical MNI outputs respect ``--output-space`` selection (#490)
* ENH: Added support for distortion correction using opposite phase encoding direction EPI images (#493)
* ENH: Switched to FSL BET for skullstripping of EPI images (#493)
* ENH: ``--omp-nthreads`` controls maximum per-process thread count; replaces ``--ants-nthreads`` (#500)

0.4.1 (20th of April 2017)
--------------------------
* Hotfix release (dependencies and deployment system)

0.4.0 (20th of April 2017)
--------------------------
* ENH: Added an option to choose the degrees of freedom used when doing BOLD to T1w coregistration (``--bold2t1w_dof``). Set default to 9 to account for field inhomogeneities and coils heating up (#448)
* ENH: Added support for phase difference and GE style fieldmaps (#448)
* ENH: Generate GrayWhite, Pial, MidThickness and inflated surfaces (#398)
* ENH: Memory and performance improvements for calculating the EPI reference (#436)
* ENH: Sample functional series to subject and ``fsaverage`` surfaces (#391)
* ENH: Output spaces for functional data may be selected with ``--output-space`` option (#447)
* ENH: ``--skip-native`` functionality replaced by ``--output-space`` (#447)
* ENH: ``fmriprep-docker`` wrapper script simplifies running in a Docker environment (#317)

0.3.2 (7th of April 2017)
-------------------------
With thanks to Asier Erramuzpe for contributions.

* ENH: Added optional slice time correction (#415)
* ENH: Removed redundant motion parameter conversion step using avscale (#415)
* ENH: FreeSurfer submillimeter reconstruction may be disabled with ``--no-submm-recon`` (#422)
* ENH: Switch bbregister init from ``fsl`` to ``coreg`` (FreeSurfer native #423)
* ENH: Motion estimation now uses a smart reference image that takes advantage of T1 saturation (#421)
* FIX: Fix report generation with ``--reports-only`` (#427)

0.3.1 (24th of March 2017)
--------------------------
* ENH: Perform bias field correction of EPI images prior to coregistration (#409)
* FIX: Fix an orientation issue affecting some datasets when bbregister was used (#408)
* ENH: Minor improvements to the reports aesthetics (#428)

0.3.0 (20th of March 2017)
--------------------------
* FIX: Affine and warp MNI transforms are now applied in the correct order
* ENH: Added preliminary support for reconstruction of cortical surfaces using FreeSurfer
* ENH: Switched to bbregister for BOLD to T1 coregistration
* ENH: Switched to sinc interpolation of preprocessed BOLD and T1w outputs
* ENH: Preprocessed BOLD volumes are now saved in the T1w space instead of mean BOLD
* FIX: Fixed a bug with MCFLIRT interpolation inducing slow drift
* ENH: All files are now saved in Float32 instead of Float64 to save space

0.2.0 (13th of January 2017)
----------------------------
* Initial public release

0.1.2 (3rd of October 2016)
---------------------------
* FIX: Downloads from OSF, remove data downloader (now in niworkflows)
* FIX: pybids was missing in the install_requires
* FIX: Deprecated ``-S``/``--subject-id`` tag
* ENH: Accept subjects with several T1w images (#114)
* ENH: Documentation updates (#130, #131)
* TST: Re-enabled CircleCI tests on one subject from ds054 of OpenfMRI
* ENH: Add C3D to docker image, updated poldracklab hub (#128, #119)
* ENH: CLI is now BIDS-Apps compliant (#123)

0.1.1 (30th of July 2016)
-------------------------
* ENH: Grabbit integration (#113)
* ENH: More outputs in MNI space (#99)
* ENH: Implementation of phase-difference fieldmap estimation (#91)
* ENH: Fixed bug using non-RAS EPI
* ENH: Works on ds005 (datasets without fieldmap nor sbref)
* ENH: Outputs start to follow BIDS-derivatives (WIP)

0.0.1
-----
* ENH: Added Docker images
* DOC: Added base code for automatic publication to RTD.
* Set up CircleCI with a first smoke test on one subject.
* BIDS tree scrubbing and subject-session-run selection.
* Refactored big workflow into consistent pieces.
* Migrated Craig's original code
