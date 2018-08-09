1.1.4 (August 06, 2018)
=======================

  * [FIX] Verify first word of _cmd in dependency check (#1236)
  * [DOC] Add two missing references (#1234)
  * [ENH] Allow turning off random seeding for ANTs brain extraction (#919)

1.1.3 (July 30, 2018)
=====================

* [ENH] Use upstream ``afni.TShift`` improvements (#1160)
* [PIN] Nipype 1.1.1 (65078c9)
* [ENH] Dynamic citation boilerplate (#1024)
* [ENH] Check Command Line dependencies before running (#1044)
* [ENH] Reimplement ``SignalExtraction`` (#1170)
* [DOC] Update copyright year to 2018 (#1224)
* [ENH] Enable ``-u`` (docker user/userid) flag in wrapper (#1223)
* [FIX] Corrects Dockerfile ``WORKDIR``. (#1218)
* [ENH] More specific errors for missing echo times (#1221)
* [ENH] Change ``WORKDIR`` of Docker image (#1204)
* [DOC] Update documentation related to contributions (#1187)
* [DOC] Additions to include before responding to reviews of the pre-print (#1195)
* [DOC] Improving documentation on using Singularity (#1063)
* [DOC] Add OHBM 2018 poster, presentation (#1198)
* [ENH] Replace ``InvertT1w`` with upstream ``Rescale(invert=True)`` (#1161)

1.1.2 (July 6, 2018)
====================

This release incorporates Nipype improvements that should reduce the
chance of hanging if tasks are killed for excessive resource consumption.

Thanks to Elizabeth DuPre for documentation updates.

* [DOC] Clarify how to reuse FreeSurfer derivatives (#1189)
* [DOC] Improve command line option documentation (#1186, #1080)
* [MAINT] Update core dependencies (#1179, #1180)

1.1.1 (June 7, 2018)
====================

* [ENH] Pre-cache DKT31 template in Docker image (#1159)
* [MAINT] Update core dependencies (#1163)

1.1.0 (June 4, 2018)
====================

* [ENH] Use Reorient interface included upstream in nipype (#1153)
* [FIX] Refine BIDS queries to avoid indexing derivatives (#1141)
* [DOC] Clarify outlier columns (#1138)
* [PIN] Update to niworkflows 0.4.0 and nipype 1.0.4 (#1133)

1.0.15 (May 17, 2018)
=====================

* [DOC] Add lesion masking during registration (#1113)
* [FIX] Patch ``boldbuffer`` for ME (#1134)

1.0.14 (May 15, 2018)
=====================

With thanks to @ZhifangYe for contributions

* [FIX] Non-invertible transforms bringing parcellation to BOLD (#1130)
* [FIX] Bad connection for ``--medial-surface-nan`` option (#1128)

1.0.13 (May 11, 2018)
=====================

With thanks to @danlurie for the outstanding contribution of #1106

* [ENH] Some nit picks on reports (#1123)
* [ENH] Carpetplot + confounds plot (#1114)
* [ENH] Add constrained cost-function masking to T1-MNI registration (#1106)
* [FIX] Circular dependency (#1104)
* [ENH] Set ``PYTHONNOUSERSITE`` in containers (#1103)


1.0.12 (May 03, 2018)
=====================

* [MAINT] fmriprep-docker: Ensure data/output/work paths are absolute (#1089)
* [ENH] Add usage tracking and centralized error reporting (#1088)
* [FIX] Ensure one motion IC index is loaded as list (#1096)
* [TST] Refactoring CircleCI setup (#1098)
* [FIX] Compression in DataSinks (#1095)
* [MAINT] fmriprep-docker: Support Python 2/3 without future or other helpers (#1082)
* [MAINT] Update npm to 10.x (#1087)
* [DOC] Prefer pre-print over Zenodo doi in boilerplate (#1086)
* [DOC] Stylistic fix (\`'template'\`) (#1083)
* [FIX] Run ICA-AROMA in ``MNI152Lin`` 2mm resampling grid (91x109x91 vox) (#1064)
* [MAINT] Remove cwebp to revert to png (#1081)
* [ENH] Allow changing the dimensionality of Melodic for AROMA. (#1052)
* [FIX] Derivatives datasink handling of compression (#1077)
* [FIX] Check for invalid sform matrices (#1072)
* [FIX] Check exit code from subprocess (#1073)
* [DOC] Add preprint fig. 1 to About (#1070)
* [FIX] Always strip session from T1w for derivative naming (#1071)
* [DOC] Add RRIDs in the citation boilerplate (#1061)
* [ENH] Generate CIFTI derivatives (#1001)


1.0.11 (April 16, 2018)
=======================

* [FIX] Do not detrend CSF/WhiteMatter/GlobalSignal (#1058)

1.0.10 (April 16, 2018)
=======================

* [TST] Re-run ds005 with only one BOLD run (#1048)
* [FIX] Patch subject_summary in reports (#1047)

1.0.9 (April 10, 2018)
======================

With thanks to @danlurie for contributions.

* [FIX] Connect inputnode to SDC for pepolar images (#1046)
* [FIX] Pass ``ref_file`` to STC check (#1038)
* [DOC] Add BBR fallback to user docs. (#1036)
* [ENH] Revise resampling grid for template outputs (#1040)
* [MAINT] DataSinks within their workflows (#1021)
* [ENH] Add FLAIR pial refinement support (#829)
* [MAINT] Upgrade to pybids 0.5 (#1027)
* [MAINT] Refactor fieldmap heuristics (#1017)
* [FIX] Use metadata to select shortest echo as ref_file (#1018)
* [ENH] Adopt versioneer to compose version names (#1007)
* [ENH] Handle first echo separately for ME-EPI (#891)


1.0.8 (February 22, 2018)
=========================

With thanks to @mgxd and @naveau for contributions.

* [FIX] ROIs Plot and output brain masks consistency (#1002)
* [FIX] Init flirt with qform (#1003)
* [DOC] Prepopulate tag when posting neurostars questions. (#987)
* [FIX] Update fmap.py : import _get_pe_index in get_ees (#984)
* [FIX] Argparse action (#985)

1.0.7 (February 13, 2018)
=========================

* [ENH] Output ``aseg`` and ``aparc`` in T1w and BOLD spaces (#957)
* [FIX] Write latest BOLD mask out (space-T1w) (#978)
* [PIN] Updating niworkflows to 0.3.1 (#962)
* [FIX] Robuster BOLD mask (#966)

1.0.6 (29th of January 2018)
============================

* [FIX] Bad connection in phasediff-fieldmap workflow (#950)
* [PIN] niworkflows-0.3.1-dev (including Nipype 1.0.0!)
* [ENH] Migrate to CircleCI 2.0 and workflows (#943)
* [ENH] Improvements to CLIs (native & wrapper) (#944)
* [FIX] Rerun tCompCor interface in case of MemoryError (#942)

1.0.5 (21st of January 2018)
============================

* [PIN] niworkflows-0.2.8 to fix several execution issues.
* [ENH] Code cleanup (#938)

1.0.4 (15th of January 2018)
============================

* [FIX] Pin niworkflows-0.2.6 to fix several MultiProc errors (nipy/nipype#2368)
* [DOC] Fix DOI in citation boilerplate (#933)
* [FIX] Heuristics to prevent memory errors during aCompCor (#930).
* [FIX] RuntimeWarning: divide by zero encountered in float_scalars (#931).
* [FIX] INU correction before merging several T1w (#925).


1.0.3 (3rd of January 2018)
===========================

* [FIX] Pin niworkflows-0.2.4 to fix (#868).
* [FIX] Roll back run/task groupings after BIDS query (#918).
  Groupings for the multi-echo extension will be reenabled soon.

1.0.2 (2nd of January 2018)
===========================

* [FIX] Grouping runs broke FMRIPREP on some datasets (#916)
  Thanks to @emdupre


1.0.1 (1st of January 2018)
===========================

With thanks to @emdupre for contributions.

* [PIN] Update required niworkflows version to 0.2.3
* [FIX] Refine ``antsBrainExtraction`` if ``recon-all`` is run (#912)
  With thanks to Arno Klein for his [helpful comments
  here](https://github.com/poldracklab/fmriprep/issues/431#issuecomment-299583391)
* [FIX] Use thinner contours in reportlets (#910)
* [FIX] Robuster EPI mask (#911)
* [FIX] Set workflow return value before potential error (#887)
* [DOC] Documentation about FreeSurfer and ``--fs-no-reconall`` (#894)
* [DOC] Fix example in installation ants-nthreads -> omp-nthreads (#885)
  With thanks to @mvdoc.
* [ENH] Allow for multiecho data (#875)


1.0.0 (6th of December 2017)
============================

* [ENH] Add ``--resource-monitor`` flag (#883)
* [FIX] Collision between Multi-T1w and ``--no-freesurfer`` (#880)
* [FIX] Setting ``use_compression`` on resampling workflows (#882)
* [ENH] Estimate motion parameters before STC (#876)
* [ENH] Add ``--stop-on-first-crash`` option (#865)
* [FIX] Correctly handling xforms (#874)
* [FIX] Combined ROI reportlets (#872)
* [ENH] Strip reportlets out of full report (#867)

1.0.0-rc13 (1st of December 2017)
---------------------------------

* [FIX] Broken ``--fs-license-file`` argument (#869)

1.0.0-rc12 (29th of November 2017)
----------------------------------

* [ENH] Use Nipype MultiProc even for sequential execution (#856)
* [REF] More memory annotations and considerations (#816)
* [FIX] Controlling memory explosion (#854)
* [WRAPPER] Mount nipype repositories as niworkflows submodule (#834)
* [FIX] Reduce image loads in local memory (#839)
* [ENH] Always sync qforms, refactor error messaging (#851)

1.0.0-rc11 (24th of November 2017)
----------------------------------

* [ENH] Check for invalid qforms in validation (#847)
* [FIX] Update pybids to include latest bugfixes (#838)
* [FIX] MultiApplyTransforms failed with nthreads=1 (#835)

1.0.0-rc10 (9th of November 2017)
---------------------------------

* [FIX] Adopt new FreeSurfer (v6.0.1) license mechanism (#787)
* [ENH] Output affine transforms from original T1w images to preprocessed anatomical (#726)
* [FIX] Correct headers in AFNI-generated NIfTI files (#818)
* [FIX] Normalize T1w image qform/sform matrices (#820)

1.0.0-rc9 (2nd of November 2017)
--------------------------------

* [FIX] Fixed #776 (aCompCor - numpy.linalg.linalg.LinAlgError: SVD did not converge) via #807.
* [ENH] Added ``CSF`` column to ``_confounds.tsv`` (included in #807)
* [DOC] Add more details on the outputs of FMRIPREP and minor fixes (#811)
* [ENH] Processing confounds in BOLD space (#807)
* [ENH] Updated niworkflows and nipype, including the new feature to close all file descriptors (#810)
* [REF] Refactored BOLD workflows module (#805)
* [ENH] Improved memory annotations (#803, #807)

1.0.0-rc8 (27th of October 2017)
--------------------------------

* [FIX] Allow missing magnitude2 in phasediff-type fieldmaps (#802)
* [FIX] Lower tolerance deciding t1_merge shapes (#798)
* [FIX] Be robust to 4D T1w images (#797)
* [ENH] Resource annotations (#746)
* [ENH] Use indexed_gzip with nibabel (#788)
* [FIX] Reduce FoV of outputs in T1w space (#785)


1.0.0-rc7 (20th of October 2017)
--------------------------------

* [ENH] Update pinned version of nipype to latest master
* [ENH] Added rX permissions to make life easier on Singularity users (#757)
* [DOC] Citation boilerplate (#779)
* [FIX] Patch to remove long filenames after mri_concatenate_lta (#778)
* [FIX] Only use unbiased template with ``--longitudinal`` (#771)
* [FIX] Use t1_2_fsnative registration when sampling to surface (#762)
* [ENH] Remove ``--skull_strip_ants`` option (#761)
* [DOC] Add reference to beginners guide (#763)


1.0.0-rc6 (11th of October 2017)
--------------------------------

* [ENH] Add inverse normalization transform (MNI -> T1w) to derivatives (#754)
* [ENH] Fall back to initial registration if BBR fails (#694)
* [FIX] Header and affine transform updates to resolve intermittent
  misalignments in reports (#743)
* [FIX] Register FreeSurfer template to FMRIPREP template, handling pre-run
  FreeSurfer subjects more robustly, saving affine to derivatives (#733)
* [ENH] Add OpenFMRI participant sampler command-line tool (#704)
* [ENH] For SyN-SDC, assume phase-encoding direction of A-P unless specified
  L-R (#740, #744)
* [ENH] Permit skull-stripping with NKI ANTs template (#729)
* [ENH] Erode aCompCor masks to target volume proportions, instead of fixed
  distances (#731, #732)
* [DOC] Documentation updates (#748)

1.0.0-rc5 (25th of September 2017)
----------------------------------

* [FIX] Skip slice time correction on BOLD series < 5 volumes (#711)
* [FIX] Skip AFNI check for new versions (#723)
* [DOC] Documentation clarification and updates (#698, #711)

1.0.0-rc4 (12th of September 2017)
----------------------------------

With thanks to Mathias Goncalves for contributions.

* [ENH] Collapse ITK transforms of head-motion correction in only one file (#695)
* [FIX] Raise error when run.py is called directly (#692)
* [FIX] Parse crash files when they are stored as text (#690)
* [ENH] Replace medial wall values with NaNs (#687)

1.0.0-rc3 (28th of August 2017)
-------------------------------

With thanks to Anibal Sólon for contributions.

* [ENH] Add ``--low-mem`` option to reduce memory usage for large BOLD series (#663)
* [ENH] Parallelize anatomical conformation step (#666)
* [FIX] Handle missing functional data in SubjectSummary node (#670)
* [FIX] Disable ``--no-skull-strip-ants`` (AFNI skull-stripping) (#674)
* [FIX] Initialize SyN SDC more robustly (#680)
* [DOC] Add comprehensive documentation of workflow API (#638)

1.0.0-rc2 (12th of August 2017)
-------------------------------

* [ENH] Increased support for partial field-of-view BOLD datasets (#659)
* [FIX] Slice time correction is now being applied to output data (not only to intermediate file used for motion estimation - #662)
* [FIX] Fieldmap unwarping is now being applied to MNI space outputs (not only to T1w space outputs - #662)

1.0.0-rc1 (8th of August 2017)
------------------------------

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
* [DEP] Deprecated ``-S``/``--subject-id`` tag
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
