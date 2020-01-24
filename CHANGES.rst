1.5.7 (January 23, 2020)
========================
Bug-fix release in the 1.5.x series.

This release fixes a bug specifically for T1w images with dimensions ≤256 voxels
but a field-of-view >256mm.

  * FIX: Calculate FoV with shape and zooms (poldracklab/smriprep#161)

1.5.6 (January 22, 2020)
========================
Bug-fix release in the 1.5.x series.

  * FIX: Include all functional runs in reports, establish consistent ordering (#1937)
  * FIX: Use SyN-SDC if --use-syn-sdc and --ignore fieldmaps are used (#1942)

1.5.5 (January 14, 2020)
=========================
Bug-fix release in the 1.5.x series.

* FIX: Correctly select volumetric spaces for carpetplot (#1932) @effigies
* FIX: Constrain setuptools for Python 2.7 installs of fmriprep-docker (#1933) @effigies

1.5.4 (December 18, 2019)
=========================
Bug-fix release in the 1.5.x series.

* FIX: Integrate fix for poldracklab/sdcflows#77 (pin niworkflows-1.0.3, sdcflows-1.0.3) @oesteban

1.5.3 (December 12, 2019)
=========================
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
========================
Bug-fix release in the 1.5.x series.

* FIX: Ensure data type of masked image matches T1.mgz (poldracklab/niworkflows#430) @effigies

1.5.1 (November 26, 2019)
=========================
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
=========================
Two hallmark changes conducive to a new minor release line have been included in
version 1.5.0: the upgrade of *PyBIDS* to the 0.9 series and the split of *SDCflows*
off from *fMRIPrep* codebase.
*PyBIDS* 0.9.x has a better handling of the indexed dataset that will permit some
optimizations to *fMRIPrep*'s memory fingerprint.
*SDCflows* now are found at `poldracklab/sdcflows <https://github.com/poldracklab/sdcflows>`__,
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

1.4.1 (July 9, 2019)
====================
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
====================
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

* FIX: Orientation problem with niworkflows<0.7.2 (poldracklab/niworkflows#312) @effigies
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
`#1500 <https://github.com/poldracklab/fmriprep/issues/1500>`__.

1.3.0 (February 7, 2019)
------------------------
We start the 1.3.x series including a few bugfixes, housekeeping duty and a refactors
to leverage `sMRIPrep <https://github.com/poldracklab/smriprep>`__ (which is a fork of
fMRIPrep's anatomical workflow), pybids>=0.7 for querying dataset, and
`TemplateFlow <https://github.com/templateflow>`__ for handling standard spaces.

* FIX: Bad ``fsnative`` replacement in CIfTI workflow (#1476) @oesteban
* FIX: Avoid warning when generating boilerplate (#1464) @oesteban
* MAINT: resolves #1485 : patch fmriprep-docker automount for use with Python 3.7 (#1486) @rciric
* RF: Use anatomical workflows from sMRIPrep (#1482) @oesteban
* MAINT: Update sentry-sdk (#1490) @chrisfilo
* ENH: Remaining TemplateFlow integrations (#1494) @oesteban
* MAINT: Update to keep up with poldracklab/niworkflows#299 (#1496) @oesteban
* FIX: Updating bids-validator to 1.1.3 (#1498) @chrisfilo

1.2.x series (January 2019)
===========================
1.2.6-1 (January 24, 2019)
--------------------------
Hotfix release of version 1.2.6, pinning niworkflows to a release version (instead
of the development branch, since #1459) and including to bugfixes.

* PIN: NiWorkflows 0.5.2.post7 (`1bf4a21 <https://github.com/poldracklab/fmriprep/commit/1bf4a21cce62c4330510a9a8ae50db876fbc23b0>`__).
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
* CI: Deploy poldracklab/fmriprep:unstable tracking master (#1307) @effigies

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
<https://github.com/poldracklab/fmriprep/issues/1235>`_. Additionally,
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
  here](https://github.com/poldracklab/fmriprep/issues/431#issuecomment-299583391)
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
