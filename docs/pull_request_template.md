## Changes proposed in this pull request

<!--
Please describe here the main features / changes proposed for review and integration in fMRIPrep
If this PR addresses some existing problem, please use GitHub's citing tools 
(eg. ref #, closes # or fixes #).
If there is not an existing issue open describing the problem, please consider opening a new
issue first and then link it from here (so the *fMRIPrep* community has a better understanding
of ongoing development efforts and possible overlaps between contributions).
-->


## Documentation that should be reviewed
<!--
Please summarize here the main changes to the documentation that the reviewers should be aware of.
-->

## Pull-request guidelines:
<!--
Please make sure you have read the PR guidelines below, and checked 
all boxes that apply at the bottom.
-->
1. We invite you to list yourself as a *fMRIPrep* contributor, so if your name 
   is not already mentioned, please modify the 
   [``.zenodo.json``](https://github.com/poldracklab/fmriprep/blob/master/.zenodo.json)
   file with your data right above Russ' entry. Example:
   ```
   {
      "name": "Contributor, New FMRIPrep",
      "affiliation": "Department of fMRI prep'ing, Open Science Made-Up University",
      "orcid": "<your id>"
   },
   {
      "name": "Poldrack, Russell A.",
      "affiliation": "Department of Psychology, Stanford University",
      "orcid": "0000-0001-6755-0259"
   },
   ```
   
2. By submitting this request you acknowledge that your contributions are available under the BSD 3-Clause license.

3. Use a descriptive prefix, between brackets for your PR: ``ENH`` (enhancement), ``FIX``, ``TST``, ``DOC``, ``STY``,
   ``REF`` (refactor), ``CI`` (continous integration), ``MAINT`` (maintenance). Example:
   ```
   [ENH] Support for SB-reference in multi-band datasets
   ```
   For works-in-progress, add the ``WIP`` tag in addition to the descriptive prefix. 
   Pull-requests tagged with ``[WIP]`` will not merged in until the tag is removed.

4. Your PR will be reviewed according to the following
   [template](https://github.com/poldracklab/fmriprep/wiki/Reviewing-a-Pull-Request).

5. Documentation is a fundamental aspect to the *glass-box* philosophy that *fMRIPrep* abides by.
   Please understand that the *fMRIPrep* team may (are likely to) request you to improve the documentation
   provided with this PR, within the PR or in future PRs.
   

**Please review and check the following**:
<!-- replace the empty checkboxes [ ] below with checked ones [x] accordingly -->

  - [ ] I have read the [guidelines for contributions](https://github.com/poldracklab/fmriprep/blob/master/CONTRIBUTING.md).
  - [ ] I understand that my contributions will not be merged unless the work is
        finished (i.e. no ``[WIP]`` tag remains in the title of my PR) and tests pass.
  - [ ] The proposed code follows the
        [coding style](https://github.com/poldracklab/fmriprep/blob/master/CONTRIBUTING.md#fmriprep-coding-style-guide),
        to the extent I understood them (and I will address any comments raised by the PR's reviewers in this regard).
  - [ ] \(Optional\) I opt-out from being listed in the `.zenodo.json` file.
