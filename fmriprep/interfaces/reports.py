#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to generate reportlets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
from niworkflows.nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiPath, isdefined)
from niworkflows.interfaces.base import SimpleInterface

from niworkflows.nipype.interfaces import freesurfer as fs

ANATOMICAL_TEMPLATE = """\t\t<h3 class="elem-title">Summary</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Structural images: {n_t1s:d}</li>
\t\t\t<li>FreeSurfer reconstruction: {freesurfer_status}</li>
\t\t\t<li>Output spaces: {output_spaces}</li>
\t\t</ul>
"""


FUNCTIONAL_TEMPLATE = """\t\t<h3 class="elem-title">Summary</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Slice timing correction: {stc}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Registration: {registration}</li>
\t\t\t<li>Functional series resampled to spaces: {output_spaces}</li>
\t\t\t<li>Confounds collected: {confounds}</li>
\t\t</ul>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = os.path.abspath('report.html')
        with open(fname, 'w') as fobj:
            fobj.write(segment)

        self._results['out_report'] = fname

        return runtime


class AnatomicalSummaryInputSpec(BaseInterfaceInputSpec):
    t1w = InputMultiPath(File(exists=True), desc='T1w structural images')
    subjects_dir = Directory(desc='FreeSurfer subjects directory')
    subject_id = traits.Str(desc='FreeSurfer subject ID')
    output_spaces = traits.List(desc='Target spaces')
    template = traits.Enum('MNI152NLin2009cAsym', desc='Template space')


class AnatomicalSummaryOutputSpec(SummaryOutputSpec):
    subject_id = traits.Str(desc='FreeSurfer subject ID')


class AnatomicalSummary(SummaryInterface):
    input_spec = AnatomicalSummaryInputSpec
    output_spec = AnatomicalSummaryOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.subject_id):
            self._results['subject_id'] = self.inputs.subject_id
        return super(AnatomicalSummary, self)._run_interface(runtime)

    def _generate_segment(self):
        if not isdefined(self.inputs.subjects_dir):
            freesurfer_status = 'Not run'
        else:
            recon = fs.ReconAll(subjects_dir=self.inputs.subjects_dir,
                                subject_id=self.inputs.subject_id,
                                T1_files=self.inputs.t1w,
                                flags='-noskullstrip')
            if recon.cmdline.startswith('echo'):
                freesurfer_status = 'Pre-existing directory'
            else:
                freesurfer_status = 'Run by FMRIPREP'

        output_spaces = [self.inputs.template if space == 'template' else space
                         for space in self.inputs.output_spaces
                         if space[:9] in ('fsaverage', 'template')]

        return ANATOMICAL_TEMPLATE.format(n_t1s=len(self.inputs.t1w),
                                          freesurfer_status=freesurfer_status,
                                          output_spaces=', '.join(output_spaces))


class FunctionalSummaryInputSpec(BaseInterfaceInputSpec):
    slice_timing = traits.Bool(False, usedefault=True, desc='Slice timing correction used')
    distortion_correction = traits.Enum('epi', 'fieldmap', 'phasediff', 'SyN', 'None',
                                        desc='Susceptibility distortion correction method',
                                        mandatory=True)
    registration = traits.Enum('FLIRT', 'bbregister', mandatory=True,
                               desc='Functional/anatomical registration method')
    output_spaces = traits.List(desc='Target spaces')
    confounds = traits.List(desc='Confounds collected')


class FunctionalSummary(SummaryInterface):
    input_spec = FunctionalSummaryInputSpec

    def _generate_segment(self):
        stc = "Applied" if self.inputs.slice_timing else "Not applied"
        sdc = {'epi': 'Phase-encoding polarity (pepolar)',
               'fieldmap': 'Direct fieldmapping',
               'phasediff': 'Phase difference',
               'SyN': 'Symmetric normalization (SyN) - no fieldmaps',
               'None': 'None'}[self.inputs.distortion_correction]
        reg = {'FLIRT': 'FLIRT with boundary-based registration (BBR) metric',
               'bbregister': 'FreeSurfer boundary-based registration (bbregister)'
               }[self.inputs.registration]
        return FUNCTIONAL_TEMPLATE.format(stc=stc, sdc=sdc, registration=reg,
                                          output_spaces=', '.join(self.inputs.output_spaces),
                                          confounds=', '.join(self.inputs.confounds))
