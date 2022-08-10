# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Interfaces to generate reportlets."""

import os
import time
import re
import logging

from collections import Counter
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiObject, Str, isdefined,
    SimpleInterface)
from niworkflows.interfaces.reportlets import base as nrb
from smriprep.interfaces.freesurfer import ReconAll


LOGGER = logging.getLogger('nipype.interface')

SUBJECT_TEMPLATE = """\
\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Structural images: {n_t1s:d} T1-weighted {t2w}</li>
\t\t<li>Functional series: {n_bold:d}</li>
{tasks}
\t\t<li>Standard output spaces: {std_spaces}</li>
\t\t<li>Non-standard output spaces: {nstd_spaces}</li>
\t\t<li>FreeSurfer reconstruction: {freesurfer_status}</li>
\t</ul>
"""

FUNCTIONAL_TEMPLATE = """\
\t\t<details open>
\t\t<summary>Summary</summary>
\t\t<ul class="elem-desc">
\t\t\t<li>Original orientation: {ornt}</li>
\t\t\t<li>Repetition time (TR): {tr:.03g}s</li>
\t\t\t<li>Phase-encoding (PE) direction: {pedir}</li>
\t\t\t<li>{multiecho}</li>
\t\t\t<li>Slice timing correction: {stc}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Registration: {registration}</li>
\t\t\t<li>Non-steady-state volumes: {dummy_scan_desc}</li>
\t\t</ul>
\t\t</details>
\t\t<details>
\t\t\t<summary>Confounds collected</summary><br />
\t\t\t<p>{confounds}.</p>
\t\t</details>
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>fMRIPrep version: {version}</li>
\t\t<li>fMRIPrep command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = os.path.join(runtime.cwd, 'report.html')
        with open(fname, 'w') as fobj:
            fobj.write(segment)
        self._results['out_report'] = fname
        return runtime

    def _generate_segment(self):
        raise NotImplementedError


class SubjectSummaryInputSpec(BaseInterfaceInputSpec):
    t1w = InputMultiObject(File(exists=True), desc='T1w structural images')
    t2w = InputMultiObject(File(exists=True), desc='T2w structural images')
    subjects_dir = Directory(desc='FreeSurfer subjects directory')
    subject_id = Str(desc='Subject ID')
    bold = InputMultiObject(traits.Either(
        File(exists=True), traits.List(File(exists=True))),
        desc='BOLD functional series')
    std_spaces = traits.List(Str, desc='list of standard spaces')
    nstd_spaces = traits.List(Str, desc='list of non-standard spaces')


class SubjectSummaryOutputSpec(SummaryOutputSpec):
    # This exists to ensure that the summary is run prior to the first ReconAll
    # call, allowing a determination whether there is a pre-existing directory
    subject_id = Str(desc='FreeSurfer subject ID')


class SubjectSummary(SummaryInterface):
    input_spec = SubjectSummaryInputSpec
    output_spec = SubjectSummaryOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.subject_id):
            self._results['subject_id'] = self.inputs.subject_id
        return super(SubjectSummary, self)._run_interface(runtime)

    def _generate_segment(self):
        BIDS_NAME = re.compile(
            r'^(.*\/)?'
            '(?P<subject_id>sub-[a-zA-Z0-9]+)'
            '(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
            '(_(?P<task_id>task-[a-zA-Z0-9]+))?'
            '(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
            '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
            '(_(?P<run_id>run-[a-zA-Z0-9]+))?')

        if not isdefined(self.inputs.subjects_dir):
            freesurfer_status = 'Not run'
        else:
            recon = ReconAll(subjects_dir=self.inputs.subjects_dir,
                             subject_id='sub-' + self.inputs.subject_id,
                             T1_files=self.inputs.t1w,
                             flags='-noskullstrip')
            if recon.cmdline.startswith('echo'):
                freesurfer_status = 'Pre-existing directory'
            else:
                freesurfer_status = 'Run by fMRIPrep'

        t2w_seg = ''
        if self.inputs.t2w:
            t2w_seg = '(+ {:d} T2-weighted)'.format(len(self.inputs.t2w))

        # Add list of tasks with number of runs
        bold_series = self.inputs.bold if isdefined(self.inputs.bold) else []
        bold_series = [s[0] if isinstance(s, list) else s for s in bold_series]

        counts = Counter(BIDS_NAME.search(series).groupdict()['task_id'][5:]
                         for series in bold_series)

        tasks = ''
        if counts:
            header = '\t\t<ul class="elem-desc">'
            footer = '\t\t</ul>'
            lines = ['\t\t\t<li>Task: {task_id} ({n_runs:d} run{s})</li>'.format(
                     task_id=task_id, n_runs=n_runs, s='' if n_runs == 1 else 's')
                     for task_id, n_runs in sorted(counts.items())]
            tasks = '\n'.join([header] + lines + [footer])

        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_t1s=len(self.inputs.t1w),
            t2w=t2w_seg,
            n_bold=len(bold_series),
            tasks=tasks,
            std_spaces=', '.join(self.inputs.std_spaces),
            nstd_spaces=', '.join(self.inputs.nstd_spaces),
            freesurfer_status=freesurfer_status)


class FunctionalSummaryInputSpec(BaseInterfaceInputSpec):
    slice_timing = traits.Enum(False, True, 'TooShort', usedefault=True,
                               desc='Slice timing correction used')
    distortion_correction = traits.Str(desc='Susceptibility distortion correction method',
                                       mandatory=True)
    pe_direction = traits.Enum(None, 'i', 'i-', 'j', 'j-', 'k', 'k-', mandatory=True,
                               desc='Phase-encoding direction detected')
    registration = traits.Enum('FSL', 'FreeSurfer', mandatory=True,
                               desc='Functional/anatomical registration method')
    fallback = traits.Bool(desc='Boundary-based registration rejected')
    registration_dof = traits.Enum(6, 9, 12, desc='Registration degrees of freedom',
                                   mandatory=True)
    registration_init = traits.Enum('register', 'header', mandatory=True,
                                    desc='Whether to initialize registration with the "header"'
                                         ' or by centering the volumes ("register")')
    confounds_file = File(exists=True, desc='Confounds file')
    tr = traits.Float(desc='Repetition time', mandatory=True)
    dummy_scans = traits.Either(traits.Int(), None, desc='number of dummy scans specified by user')
    algo_dummy_scans = traits.Int(desc='number of dummy scans determined by algorithm')
    echo_idx = traits.List([], usedefault=True, desc="BIDS echo identifiers")
    orientation = traits.Str(mandatory=True, desc='Orientation of the voxel axes')


class FunctionalSummary(SummaryInterface):
    input_spec = FunctionalSummaryInputSpec

    def _generate_segment(self):
        dof = self.inputs.registration_dof
        stc = {True: 'Applied',
               False: 'Not applied',
               'TooShort': 'Skipped (too few volumes)'}[self.inputs.slice_timing]
        # #TODO: Add a note about registration_init below?
        reg = {
            'FSL': [
                'FSL <code>flirt</code> with boundary-based registration'
                ' (BBR) metric - %d dof' % dof,
                'FSL <code>flirt</code> rigid registration - 6 dof'],
            'FreeSurfer': [
                'FreeSurfer <code>bbregister</code> '
                '(boundary-based registration, BBR) - %d dof' % dof,
                'FreeSurfer <code>mri_coreg</code> - %d dof' % dof],
        }[self.inputs.registration][self.inputs.fallback]

        pedir = get_world_pedir(self.inputs.orientation, self.inputs.pe_direction)

        if isdefined(self.inputs.confounds_file):
            with open(self.inputs.confounds_file) as cfh:
                conflist = cfh.readline().strip('\n').strip()

        dummy_scan_tmp = "{n_dum}"
        if self.inputs.dummy_scans == self.inputs.algo_dummy_scans:
            dummy_scan_msg = (
                ' '.join([dummy_scan_tmp, "(Confirmed: {n_alg} automatically detected)"])
                .format(n_dum=self.inputs.dummy_scans, n_alg=self.inputs.algo_dummy_scans)
            )
        # the number of dummy scans was specified by the user and
        # it is not equal to the number detected by the algorithm
        elif self.inputs.dummy_scans is not None:
            dummy_scan_msg = (
                ' '.join([dummy_scan_tmp, "(Warning: {n_alg} automatically detected)"])
                .format(n_dum=self.inputs.dummy_scans, n_alg=self.inputs.algo_dummy_scans)
            )
        # the number of dummy scans was not specified by the user
        else:
            dummy_scan_msg = dummy_scan_tmp.format(n_dum=self.inputs.algo_dummy_scans)

        multiecho = "Single-echo EPI sequence."
        n_echos = len(self.inputs.echo_idx)
        if n_echos == 1:
            multiecho = (
                f"Multi-echo EPI sequence: only echo {self.inputs.echo_idx[0]} processed "
                "in single-echo mode."
            )
        if n_echos > 2:
            multiecho = (f"Multi-echo EPI sequence: {n_echos} echoes.")

        return FUNCTIONAL_TEMPLATE.format(
            pedir=pedir, stc=stc, sdc=self.inputs.distortion_correction, registration=reg,
            confounds=re.sub(r'[\t ]+', ', ', conflist), tr=self.inputs.tr,
            dummy_scan_desc=dummy_scan_msg, multiecho=multiecho, ornt=self.inputs.orientation)


class AboutSummaryInputSpec(BaseInterfaceInputSpec):
    version = Str(desc='FMRIPREP version')
    command = Str(desc='FMRIPREP command')
    # Date not included - update timestamp only if version or command changes


class AboutSummary(SummaryInterface):
    input_spec = AboutSummaryInputSpec

    def _generate_segment(self):
        return ABOUT_TEMPLATE.format(version=self.inputs.version,
                                     command=self.inputs.command,
                                     date=time.strftime("%Y-%m-%d %H:%M:%S %z"))


class LabeledHistogramInputSpec(nrb._SVGReportCapableInputSpec):
    in_file = traits.File(
        exists=True,
        mandatory=True,
        desc="Image containing values to plot"
    )
    label_file = traits.File(
        exists=True,
        desc="Mask or label image where non-zero values will be used to extract data from in_file"
    )
    mapping = traits.Dict(desc="Map integer label values onto names of voxels")
    xlabel = traits.Str("voxels", usedefault=True, desc="Description of values plotted")


class LabeledHistogram(nrb.ReportingInterface):
    input_spec = LabeledHistogramInputSpec

    def _generate_report(self):
        import numpy as np
        import nibabel as nb
        from nilearn.image import resample_to_img
        from matplotlib import pyplot as plt
        import seaborn as sns

        report_file = self._out_report
        img = nb.load(self.inputs.in_file)
        data = img.get_fdata(dtype=np.float32)

        if self.inputs.label_file:
            label_img = nb.load(self.inputs.label_file)
            if label_img.shape != img.shape[:3] or not np.allclose(label_img.affine, img.affine):
                label_img = resample_to_img(label_img, img, interpolation="nearest")
            labels = np.uint16(label_img.dataobj)
        else:
            labels = np.uint8(data > 0)

        uniq_labels = np.unique(labels[labels > 0])
        label_map = self.inputs.mapping or {label: label for label in uniq_labels}

        rois = {
            label_map.get(label, label): data[labels == label]
            for label in label_map
        }
        with sns.axes_style('whitegrid'):
            fig = sns.histplot(rois, bins=50)
            fig.set_xlabel(self.inputs.xlabel)
        plt.savefig(report_file)
        plt.close()


def get_world_pedir(ornt, pe_direction):
    """Return world direction of phase encoding"""
    axes = (
        ("Right", "Left"),
        ("Anterior", "Posterior"),
        ("Superior", "Inferior")
    )
    ax_idcs = {"i": 0, "j": 1, "k": 2}

    if pe_direction is not None:
        axcode = ornt[ax_idcs[pe_direction[0]]]
        inv = pe_direction[1:] == "-"

        for ax in axes:
            for flip in (ax, ax[::-1]):
                if flip[not inv].startswith(axcode):
                    return "-".join(flip)
    LOGGER.warning(
        "Cannot determine world direction of phase encoding. "
        f"Orientation: {ornt}; PE dir: {pe_direction}"
    )
    return "Could not be determined - assuming Anterior-Posterior"
