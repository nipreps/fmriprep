# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetch some example data:

    >>> import os
    >>> from niworkflows import data
    >>> data_root = data.get_bids_examples(variant='BIDS-examples-1-enh-ds054')
    >>> os.chdir(data_root)

Disable warnings:

    >>> from nipype import logging
    >>> logging.getLogger('nipype.interface').setLevel('ERROR')

"""

import os
import os.path as op
import re
import simplejson as json
import gzip
from shutil import copytree, rmtree, copyfileobj

from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec,
    File, Directory, InputMultiPath, OutputMultiPath, Str,
    SimpleInterface
)
from nipype.utils.filemanip import copyfile

LOGGER = logging.getLogger('nipype.interface')
BIDS_NAME = re.compile(
    '^(.*\/)?(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
    '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
    '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?')


class FileNotFoundError(IOError):
    pass


class BIDSInfoInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='input file, part of a BIDS tree')


class BIDSInfoOutputSpec(TraitedSpec):
    subject_id = traits.Str()
    session_id = traits.Str()
    task_id = traits.Str()
    acq_id = traits.Str()
    rec_id = traits.Str()
    run_id = traits.Str()


class BIDSInfo(SimpleInterface):
    """
    Extract metadata from a BIDS-conforming filename

    This interface uses only the basename, not the path, to determine the
    subject, session, task, run, acquisition or reconstruction.

    >>> from fmriprep.interfaces import BIDSInfo
    >>> from fmriprep.utils.bids import collect_data
    >>> bids_info = BIDSInfo()
    >>> bids_info.inputs.in_file = collect_data('ds114', '01')[0]['bold'][0]
    >>> bids_info.inputs.in_file  # doctest: +ELLIPSIS
    '.../ds114/sub-01/ses-retest/func/sub-01_ses-retest_task-covertverbgeneration_bold.nii.gz'
    >>> res = bids_info.run()
    >>> res.outputs
    <BLANKLINE>
    acq_id = <undefined>
    rec_id = <undefined>
    run_id = <undefined>
    session_id = ses-retest
    subject_id = sub-01
    task_id = task-covertverbgeneration
    <BLANKLINE>

    """
    input_spec = BIDSInfoInputSpec
    output_spec = BIDSInfoOutputSpec

    def _run_interface(self, runtime):
        match = BIDS_NAME.search(self.inputs.in_file)
        params = match.groupdict() if match is not None else {}
        self._results = {key: val for key, val in list(params.items())
                         if val is not None}
        return runtime


class BIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    subject_data = traits.Dict(Str, traits.Any)
    subject_id = Str()


class BIDSDataGrabberOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='output data structure')
    fmap = OutputMultiPath(desc='output fieldmaps')
    bold = OutputMultiPath(desc='output functional images')
    sbref = OutputMultiPath(desc='output sbrefs')
    t1w = OutputMultiPath(desc='output T1w images')
    roi = OutputMultiPath(desc='output ROI images')
    t2w = OutputMultiPath(desc='output T2w images')
    flair = OutputMultiPath(desc='output FLAIR images')


class BIDSDataGrabber(SimpleInterface):
    """
    Collect files from a BIDS directory structure

    >>> from fmriprep.interfaces import BIDSDataGrabber
    >>> from fmriprep.utils.bids import collect_data
    >>> bids_src = BIDSDataGrabber(anat_only=False)
    >>> bids_src.inputs.subject_data = collect_data('ds114', '01')[0]
    >>> bids_src.inputs.subject_id = 'ds114'
    >>> res = bids_src.run()
    >>> res.outputs.t1w  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['.../ds114/sub-01/ses-retest/anat/sub-01_ses-retest_T1w.nii.gz',
     '.../ds114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz']

    """
    input_spec = BIDSDataGrabberInputSpec
    output_spec = BIDSDataGrabberOutputSpec
    _require_funcs = True

    def __init__(self, *args, **kwargs):
        anat_only = kwargs.pop('anat_only')
        super(BIDSDataGrabber, self).__init__(*args, **kwargs)
        if anat_only is not None:
            self._require_funcs = not anat_only

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results['out_dict'] = bids_dict
        self._results.update(bids_dict)

        if not bids_dict['t1w']:
            raise FileNotFoundError('No T1w images found for subject sub-{}'.format(
                self.inputs.subject_id))

        if self._require_funcs and not bids_dict['bold']:
            raise FileNotFoundError('No functional images found for subject sub-{}'.format(
                self.inputs.subject_id))

        for imtype in ['bold', 't2w', 'flair', 'fmap', 'sbref', 'roi']:
            if not bids_dict[imtype]:
                LOGGER.warn('No \'{}\' images found for sub-{}'.format(
                    imtype, self.inputs.subject_id))

        return runtime


class DerivativesDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc='Path to the base directory for storing data.')
    in_file = InputMultiPath(File(exists=True), mandatory=True,
                             desc='the object to be saved')
    source_file = File(exists=False, mandatory=True, desc='the input func file')
    suffix = traits.Str('', mandatory=True, desc='suffix appended to source_file')
    extra_values = traits.List(traits.Str)
    compress = traits.Bool(desc="force compression (True) or uncompression (False)"
                                " of the output file (default: same as input)")


class DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))
    compression = OutputMultiPath(
        traits.Bool, desc='whether ``in_file`` was compressed/uncompressed '
                          'or `it was copied directly.')


class DerivativesDataSink(SimpleInterface):
    """
    Saves the `in_file` into a BIDS-Derivatives folder provided
    by `base_directory`, given the input reference `source_file`.

    >>> import tempfile
    >>> from fmriprep.utils.bids import collect_data
    >>> tmpdir = tempfile.mkdtemp()
    >>> tmpfile = os.path.join(tmpdir, 'a_temp_file.nii.gz')
    >>> open(tmpfile, 'w').close()  # "touch" the file
    >>> dsink = DerivativesDataSink(base_directory=tmpdir)
    >>> dsink.inputs.in_file = tmpfile
    >>> dsink.inputs.source_file = collect_data('ds114', '01')[0]['t1w'][0]
    >>> dsink.inputs.suffix = 'target-mni'
    >>> res = dsink.run()
    >>> res.outputs.out_file  # doctest: +ELLIPSIS
    '.../fmriprep/sub-01/ses-retest/anat/sub-01_ses-retest_T1w_target-mni.nii.gz'

    """
    input_spec = DerivativesDataSinkInputSpec
    output_spec = DerivativesDataSinkOutputSpec
    out_path_base = "fmriprep"
    _always_run = True

    def __init__(self, out_path_base=None, **inputs):
        super(DerivativesDataSink, self).__init__(**inputs)
        self._results['out_file'] = []
        if out_path_base:
            self.out_path_base = out_path_base

    def _run_interface(self, runtime):
        src_fname, _ = _splitext(self.inputs.source_file)
        _, ext = _splitext(self.inputs.in_file[0])
        if self.inputs.compress is True and not ext.endswith('.gz'):
            ext += '.gz'
        elif self.inputs.compress is False and ext.endswith('.gz'):
            ext = ext[:-3]

        m = BIDS_NAME.search(src_fname)

        # TODO this quick and dirty modality detection needs to be implemented
        # correctly
        mod = 'func'
        if 'anat' in op.dirname(self.inputs.source_file):
            mod = 'anat'
        elif 'dwi' in op.dirname(self.inputs.source_file):
            mod = 'dwi'
        elif 'fmap' in op.dirname(self.inputs.source_file):
            mod = 'fmap'

        base_directory = runtime.cwd
        if isdefined(self.inputs.base_directory):
            base_directory = op.abspath(self.inputs.base_directory)

        out_path = '{}/{subject_id}'.format(self.out_path_base, **m.groupdict())
        if m.groupdict().get('session_id') is not None:
            out_path += '/{session_id}'.format(**m.groupdict())
        out_path += '/{}'.format(mod)

        out_path = op.join(base_directory, out_path)

        os.makedirs(out_path, exist_ok=True)

        base_fname = op.join(out_path, src_fname)

        formatstr = '{bname}_{suffix}{ext}'
        if len(self.inputs.in_file) > 1 and not isdefined(self.inputs.extra_values):
            formatstr = '{bname}_{suffix}{i:04d}{ext}'

        self._results['compression'] = []
        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(
                bname=base_fname,
                suffix=self.inputs.suffix,
                i=i,
                ext=ext)
            if isdefined(self.inputs.extra_values):
                out_file = out_file.format(extra_value=self.inputs.extra_values[i])
            self._results['out_file'].append(out_file)
            self._results['compression'].append(_copy_any(fname, out_file))
        return runtime


class ReadSidecarJSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input nifti file')
    fields = traits.List(traits.Str, desc='get only certain fields')


class ReadSidecarJSONOutputSpec(TraitedSpec):
    subject_id = traits.Str()
    session_id = traits.Str()
    task_id = traits.Str()
    acq_id = traits.Str()
    rec_id = traits.Str()
    run_id = traits.Str()
    out_dict = traits.Dict()


class ReadSidecarJSON(SimpleInterface):
    """
    A utility to find and read JSON sidecar files of a BIDS tree
    """
    expr = re.compile('^sub-(?P<subject_id>[a-zA-Z0-9]+)(_ses-(?P<session_id>[a-zA-Z0-9]+))?'
                      '(_task-(?P<task_id>[a-zA-Z0-9]+))?(_acq-(?P<acq_id>[a-zA-Z0-9]+))?'
                      '(_rec-(?P<rec_id>[a-zA-Z0-9]+))?(_run-(?P<run_id>[a-zA-Z0-9]+))?')
    input_spec = ReadSidecarJSONInputSpec
    output_spec = ReadSidecarJSONOutputSpec
    _always_run = True

    def _run_interface(self, runtime):
        metadata = get_metadata_for_nifti(self.inputs.in_file)
        output_keys = [key for key in list(self.output_spec().get().keys()) if key.endswith('_id')]
        outputs = self.expr.search(op.basename(self.inputs.in_file)).groupdict()

        for key in output_keys:
            id_value = outputs.get(key)
            if id_value is not None:
                self._results[key] = outputs.get(key)

        if isdefined(self.inputs.fields) and self.inputs.fields:
            for fname in self.inputs.fields:
                self._results[fname] = metadata[fname]
        else:
            self._results['out_dict'] = metadata

        return runtime


class BIDSFreeSurferDirInputSpec(BaseInterfaceInputSpec):
    derivatives = Directory(exists=True, mandatory=True,
                            desc='BIDS derivatives directory')
    freesurfer_home = Directory(exists=True, mandatory=True,
                                desc='FreeSurfer installation directory')
    subjects_dir = traits.Str('freesurfer', usedefault=True,
                              desc='Name of FreeSurfer subjects directory')
    spaces = traits.List(traits.Str, desc='Set of output spaces to prepare')
    overwrite_fsaverage = traits.Bool(False, usedefault=True,
                                      desc='Overwrite fsaverage directories, if present')


class BIDSFreeSurferDirOutputSpec(TraitedSpec):
    subjects_dir = traits.Directory(exists=True,
                                    desc='FreeSurfer subjects directory')


class BIDSFreeSurferDir(SimpleInterface):
    """Create a FreeSurfer subjects directory in a BIDS derivatives directory
    and copy fsaverage from the local FreeSurfer distribution.

    Output subjects_dir = ``{derivatives}/{subjects_dir}``, and may be passed to
    ReconAll and other FreeSurfer interfaces.
    """
    input_spec = BIDSFreeSurferDirInputSpec
    output_spec = BIDSFreeSurferDirOutputSpec

    def _run_interface(self, runtime):
        subjects_dir = os.path.join(self.inputs.derivatives,
                                    self.inputs.subjects_dir)
        os.makedirs(subjects_dir, exist_ok=True)
        self._results['subjects_dir'] = subjects_dir

        spaces = list(self.inputs.spaces)
        # Always copy fsaverage, for proper recon-all functionality
        if 'fsaverage' not in spaces:
            spaces.append('fsaverage')

        for space in spaces:
            # Skip non-freesurfer spaces and fsnative
            if not space.startswith('fsaverage'):
                continue
            source = os.path.join(self.inputs.freesurfer_home, 'subjects', space)
            dest = os.path.join(subjects_dir, space)
            # Finesse is overrated. Either leave it alone or completely clobber it.
            if os.path.exists(dest) and self.inputs.overwrite_fsaverage:
                rmtree(dest)
            if not os.path.exists(dest):
                copytree(source, dest)

        return runtime


def get_metadata_for_nifti(in_file):
    """Fetch metadata for a given nifti file"""
    in_file = op.abspath(in_file)

    fname, ext = op.splitext(in_file)
    if ext == '.gz':
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext

    side_json = fname + '.json'
    fname_comps = op.basename(side_json).split("_")

    session_comp_list = []
    subject_comp_list = []
    top_comp_list = []
    ses = None
    sub = None

    for comp in fname_comps:
        if comp[:3] != "run":
            session_comp_list.append(comp)
            if comp[:3] == "ses":
                ses = comp
            else:
                subject_comp_list.append(comp)
                if comp[:3] == "sub":
                    sub = comp
                else:
                    top_comp_list.append(comp)

    if any([comp.startswith('ses') for comp in fname_comps]):
        bids_dir = '/'.join(op.dirname(in_file).split('/')[:-3])
    else:
        bids_dir = '/'.join(op.dirname(in_file).split('/')[:-2])

    top_json = op.join(bids_dir, "_".join(top_comp_list))
    potential_json = [top_json]

    subject_json = op.join(bids_dir, sub, "_".join(subject_comp_list))
    potential_json.append(subject_json)

    if ses:
        session_json = op.join(bids_dir, sub, ses, "_".join(session_comp_list))
        potential_json.append(session_json)

    potential_json.append(side_json)

    merged_param_dict = {}
    for json_file_path in potential_json:
        if op.isfile(json_file_path):
            with open(json_file_path, 'r') as jsonfile:
                param_dict = json.load(jsonfile)
                merged_param_dict.update(param_dict)

    return merged_param_dict


def _splitext(fname):
    fname, ext = op.splitext(op.basename(fname))
    if ext == '.gz':
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext
    return fname, ext


def _copy_any(src, dst):
    src_isgz = src.endswith('.gz')
    dst_isgz = dst.endswith('.gz')
    if src_isgz == dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    dst_open = gzip.open if dst_isgz else open
    with src_open(src, 'rb') as f_in:
        with dst_open(dst, 'wb') as f_out:
            copyfileobj(f_in, f_out)
    return True
