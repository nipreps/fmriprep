Next release
============


1.0.0-rc3 (28th of August 2017)
===============================

With thanks to Anibal Sólon for contributions.

* [ENH] Add --low-mem option to reduce memory usage for large BOLD series (#663)
* [ENH] Parallelize anatomical conformation step (#666)
* [FIX] Handle missing functional data in SubjectSummary node (#670)
* [FIX] Disable --no-skull-strip-ants (AFNI skull-stripping) (#674)
* [FIX] Initialize SyN SDC more robustly (#680)
* [DOC] Add comprehensive documentation of workflow API (#638)

1.0.0-rc2 (12th of August 2017)
===============================

* [ENH] Increased support for partial field-of-view BOLD datasets (#659)
* [FIX] Slice time correction is now being applied to output data (not only to intermediate file used for motion estimation - #662)
* [FIX] Fieldmap unwarping is now being applied to MNI space outputs (not only to T1w space outputs - #662)

1.0.0-rc1 (8th of August 2017)
==============================

* [ENH] Include ICA-AROMA confounds in report (#646)
* [ENH] Save non-aggressively denoised BOLD series (#648)
* [ENH] Improved logging messages (#621)
* [ENH] Improved resource management (#622, #629, #640, #641)
* [ENH] Improved confound header names (#634)
* [FIX] Ensure multi-T1w image datasets have RAS-oriented template (#637)
* [FIX] More informative errors for conflicting options (#632)
* [DOC] Improved report summaries (#647)

0.6.0 (31st of July 2017)
=========================

With thanks to Yaroslav Halchenko and Ilkay Isik for contributions.

* [ENH] Set threshold on up-sampling ratio in conformation, report results (#601)
* [ENH] Censor non-steady-state volumes prior to CompCor (#603)
* [FIX] Conformation failure in thick-slice, oblique T1w datasets (#601)
* [FIX] Crash/report failure of phase-difference SDC pipeline (#602, #604)
* [FIX] Prevent AFNI NIfTI extensions from crashing reference EPI estimation (#619)
* [DOC] Save logs to output directory (#605)
* [ENH] Upgrade to ICA-AROMA 0.4.1-beta (#611)

0.5.4 (20th of July 2017)
=========================

* [DOC] Improved report summaries describing steps taken (#584)
* [ENH] Uniformize command-line argument style (#592)

0.5.3 (18th of July 2017)
=========================

With thanks to Yaroslav Halchenko for contributions.

* [ENH] High-pass filter time series prior to CompCor (#577)
* [ENH] Validate and minimally conform BOLD images (#581)
* [FIX] Bug that prevented PE direction estimation (#586)
* [DOC] Log version/time in report (#587)

0.5.2 (30th of June 2017)
=========================

With thanks to James Kent for contributions.

* [ENH] Calculate noise components in functional data with ICA-AROMA (#539)
* [FIX] Remove unused parameters from function node, resolving crash (#576)

0.5.1 (24th of June 2017)
=========================

* [FIX] Invalid parameter in ``bbreg_wf`` (#572)

0.5.0 (21st of June 2017)
=========================

With thanks to James Kent for contributions.

* [ENH] EXPERIMENTAL: Fieldmap-less susceptibility correction with ``--use-syn-sdc`` option (#544)
* [FIX] Reduce interpolation artifacts in ConformSeries (#564)
* [FIX] Improve consistency of handling of fieldmaps (#565)
* [FIX] Apply T2w pial surface refinement at correct stage of FreeSurfer pipeline (#568)
* [ENH] Add ``--anat-only`` workflow option (#560)
* [FIX] Output all tissue class/probability maps (#569)
* [ENH] Upgrade to ANTs 2.2.0 (#561)

0.4.6 (14th of June 2017)
=========================

* [ENH] Conform and minimally resample multiple T1w images (#545)
* [FIX] Return non-zero exit code on all errors (#554)
* [ENH] Improve error reporting for missing subjects (#558)

0.4.5 (12th of June 2017)
=========================

With thanks to Marcel Falkiewicz for contributions.

* [FIX] Correctly display help in ``fmriprep-docker`` (#533)
* [FIX] Avoid invalid symlinks when running FreeSurfer (#536)
* [ENH] Improve dependency management for users unable to use Docker/Singularity containers (#549)
* [FIX] Return correct exit code when a Function node fails (#554)

0.4.4 (20th of May 2017)
========================

With thanks to Feilong Ma for contributions.

* [ENH] Option to provide a custom reference grid image (``--output-grid-reference``) for determining the field of view and resolution of output images (#480)
* [ENH] Improved EPI skull stripping and tissue contrast enhancements (#519)
* [ENH] Improve resource use estimates in FreeSurfer workflow (#506)
* [ENH] Moved missing values in the DVARS* and FramewiseDisplacement columns of the _confounds.tsv from last row to the first row (#523)
* [ENH] More robust initialization of the normalization procedure (#529)

0.4.3 (10th of May 2017)
========================

* [ENH] ``--output-space template`` targets template specified by ``--template`` flag (``MNI152NLin2009cAsym`` supported) (#498)
* [FIX] Fix a bug causing small numerical discrepancies in input data voxel size to lead to different FOV of the output files (#513)

0.4.2 (3rd of May 2017)
=======================

* [ENH] Use robust template generation for multiple T1w images (#481)
* [ENH] Anatomical MNI outputs respect ``--output-space`` selection (#490)
* [ENH] Added support for distortion correction using opposite phase encoding direction EPI images (#493)
* [ENH] Switched to FSL BET for skullstripping of EPI images (#493)
* [ENH] ``--omp-nthreads`` controls maximum per-process thread count; replaces ``--ants-nthreads`` (#500)

0.4.1 (20th of April 2017)
==========================

* Hotfix release (dependencies and deployment system)

0.4.0 (20th of April 2017)
==========================

* [ENH] Added an option to choose the degrees of freedom used when doing BOLD to T1w coregistration (``--bold2t1w_dof``). Set default to 9 to account for field inhomogeneities and coils heating up (#448)
* [ENH] Added support for phase difference and GE style fieldmaps (#448)
* [ENH] Generate GrayWhite, Pial, MidThickness and inflated surfaces (#398)
* [ENH] Memory and performance improvements for calculating the EPI reference (#436)
* [ENH] Sample functional series to subject and ``fsaverage`` surfaces (#391)
* [ENH] Output spaces for functional data may be selected with ``--output-space`` option (#447)
* [DEP] ``--skip-native`` functionality replaced by ``--output-space`` (#447)
* [ENH] ``fmriprep-docker`` wrapper script simplifies running in a Docker environment (#317)

0.3.2 (7th of April 2017)
=========================

With thanks to Asier Erramuzpe for contributions.

* [ENH] Added optional slice time correction (#415)
* [ENH] Removed redundant motion parameter conversion step using avscale (#415)
* [ENH] FreeSurfer submillimeter reconstruction may be disabled with ``--no-submm-recon`` (#422)
* [ENH] Switch bbregister init from ``fsl`` to ``coreg`` (FreeSurfer native #423)
* [ENH] Motion estimation now uses a smart reference image that takes advantage of T1 saturation (#421)
* [FIX] Fix report generation with ``--reports-only`` (#427)

0.3.1 (24th of March 2017)
==========================

* [ENH] Perform bias field correction of EPI images prior to coregistration (#409)
* [FIX] Fix an orientation issue affecting some datasets when bbregister was used (#408)
* [ENH] Minor improvements to the reports aesthetics (#428)

0.3.0 (20th of March 2017)
==========================

* [FIX] Affine and warp MNI transforms are now applied in the correct order
* [ENH] Added preliminary support for reconstruction of cortical surfaces using FreeSurfer
* [ENH] Switched to bbregister for BOLD to T1 coregistration
* [ENH] Switched to sinc interpolation of preprocessed BOLD and T1w outputs
* [ENH] Preprocessed BOLD volumes are now saved in the T1w space instead of mean BOLD
* [FIX] Fixed a bug with MCFLIRT interpolation inducing slow drift
* [ENH] All files are now saved in Float32 instead of Float64 to save space

0.2.0 (13th of January 2017)
============================

* Initial public release


0.1.2 (3rd of October 2016)
===========================

* [FIX] Downloads from OSF, remove data downloader (now in niworkflows)
* [FIX] pybids was missing in the install_requires
* [DEP] Deprecated -S/--subject-id tag
* [ENH] Accept subjects with several T1w images (#114)
* [ENH] Documentation updates (#130, #131)
* [TST] Re-enabled CircleCI tests on one subject from ds054 of OpenfMRI
* [ENH] Add C3D to docker image, updated poldracklab hub (#128, #119)
* [ENH] CLI is now BIDS-Apps compliant (#123)


0.1.1 (30th of July 2016)
=========================

* [ENH] Grabbit integration (#113)
* [ENH] More outputs in MNI space (#99)
* [ENH] Implementation of phase-difference fieldmap estimation (#91)
* [ENH] Fixed bug using non-RAS EPI
* [ENH] Works on ds005 (datasets without fieldmap nor sbref)
* [ENH] Outputs start to follow BIDS-derivatives (WIP)


0.0.1
=====

* [ENH] Added Docker images
* [DOC] Added base code for automatic publication to RTD.
* Set up CircleCI with a first smoke test on one subject.
* BIDS tree scrubbing and subject-session-run selection.
* Refactored big workflow into consistent pieces.
* Migrated Craig's original code
