Next release
============

* [ENH] Generate GrayWhite, Pial, MidThickness and inflated surfaces (#398)
* [ENH] Memory and performance improvements for calculating the EPI reference (#436)
* [ENH] Sample functional series to subject and ``fsaverage`` surfaces (#391)

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
