Release 0.1.2
=============

* [ENH] A first stab at fieldmaps (#247)
* [FIX] Downloads from OSF, remove data downloader (now in niworkflows)
* [FIX] pybids was missing in the install_requires
* [DEP] Deprecated -S/--subject-id tag
* [ENH] Accept subjects with several T1w images (#114)
* [ENH] Documentation updates (#130, #131)
* [TST] Re-enabled CircleCI tests on one subject from ds054 of OpenfMRI
* [ENH] Add C3D to docker image, updated poldracklab hub (#128, #119)
* [ENH] CLI is now BIDS-Apps compliant (#123)


Release 0.1.1
=============

* [ENH] Grabbit integration (#113)
* [ENH] More outputs in MNI space (#99)
* [ENH] Implementation of phase-difference fieldmap estimation (#91)
* [ENH] Fixed bug using non-RAS EPI
* [ENH] Works on ds005 (datasets without fieldmap nor sbref)
* [ENH] Outputs start to follow BIDS-derivatives (WIP)


Release 0.0.1
=============

* [ENH] Added Docker images
* [DOC] Added base code for automatic publication to RTD.
* Set up CircleCI with a first smoke test on one subject.
* BIDS tree scrubbing and subject-session-run selection.
* Refactored big workflow into consistent pieces.
* Migrated Craig's original code
