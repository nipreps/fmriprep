"""BIDS-related interfaces."""

from pathlib import Path

from bids.utils import listify
from nipype.interfaces.base import (
    DynamicTraitedSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.io import add_traits
from nipype.interfaces.utility.base import _ravel

from ..utils.bids import _find_nearest_path


class _BIDSURIInputSpec(DynamicTraitedSpec):
    dataset_links = traits.Dict(mandatory=True, desc='Dataset links')
    out_dir = traits.Str(mandatory=True, desc='Output directory')


class _BIDSURIOutputSpec(TraitedSpec):
    out = traits.List(
        traits.Str,
        desc='BIDS URI(s) for file',
    )


class BIDSURI(SimpleInterface):
    """Convert input filenames to BIDS URIs, based on links in the dataset.

    This interface can combine multiple lists of inputs.
    """

    input_spec = _BIDSURIInputSpec
    output_spec = _BIDSURIOutputSpec

    def __init__(self, numinputs=0, **inputs):
        super().__init__(**inputs)
        self._numinputs = numinputs
        if numinputs >= 1:
            input_names = [f'in{i + 1}' for i in range(numinputs)]
        else:
            input_names = []
        add_traits(self.inputs, input_names)

    def _run_interface(self, runtime):
        inputs = [getattr(self.inputs, f'in{i + 1}') for i in range(self._numinputs)]
        in_files = listify(inputs)
        in_files = _ravel(in_files)
        # Remove undefined inputs
        in_files = [f for f in in_files if isdefined(f)]
        # Convert the dataset links to BIDS URI prefixes
        updated_keys = {f'bids:{k}:': Path(v) for k, v in self.inputs.dataset_links.items()}
        updated_keys['bids::'] = Path(self.inputs.out_dir)
        # Convert the paths to BIDS URIs
        out = [_find_nearest_path(updated_keys, f) for f in in_files]
        self._results['out'] = out

        return runtime


class _BIDSSourceFileInputSpec(TraitedSpec):
    bids_info = traits.Dict(
        mandatory=True,
        desc='BIDS information dictionary',
    )
    precomputed = traits.Dict({}, usedefault=True, desc='Precomputed BIDS information')
    sessionwise = traits.Bool(False, usedefault=True, desc='Keep session information')
    anat_type = traits.Enum('t1w', 't2w', usedefault=True, desc='Anatomical reference type')


class _BIDSSourceFileOutputSpec(TraitedSpec):
    source_file = File(desc='Source file')


class BIDSSourceFile(SimpleInterface):
    input_spec = _BIDSSourceFileInputSpec
    output_spec = _BIDSSourceFileOutputSpec

    def _run_interface(self, runtime):
        src = self.inputs.bids_info[self.inputs.anat_type]

        if not src and self.inputs.precomputed.get(f'{self.inputs.anat_type}_preproc'):
            src = self.inputs.bids_info['bold']
            self._results['source_file'] = _create_multi_source_file(src)
            return runtime

        self._results['source_file'] = _create_multi_source_file(
            src,
            sessionwise=self.inputs.sessionwise,
        )
        return runtime


class _CreateFreeSurferIDInputSpec(TraitedSpec):
    subject_id = traits.Str(mandatory=True, desc='BIDS Subject ID')
    session_id = traits.Str(desc='BIDS session ID')


class _CreateFreeSurferIDOutputSpec(TraitedSpec):
    subject_id = traits.Str(desc='FreeSurfer subject ID')


class CreateFreeSurferID(SimpleInterface):
    input_spec = _CreateFreeSurferIDInputSpec
    output_spec = _CreateFreeSurferIDOutputSpec

    def _run_interface(self, runtime):
        self._results['subject_id'] = _create_fs_id(
            self.inputs.subject_id,
            self.inputs.session_id or None,
        )
        return runtime


def _create_multi_source_file(in_files, sessionwise=False):
    """
    Create a generic source name from multiple input files.

    If sessionwise is True, session information from the first file is retained in the name.

    Examples
    --------
    >>> _create_multi_source_file([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'
    >>> _create_multi_source_file([
    ...     '/path/to/sub-045_ses-1_run-1_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-1_run-2_T1w.nii.gz'],
    ...     sessionwise=True)
    '/path/to/sub-045_ses-1_T1w.nii.gz'
    """
    import re
    from pathlib import Path

    from nipype.utils.filemanip import filename_to_list

    if not isinstance(in_files, tuple | list):
        return in_files
    elif len(in_files) == 1:
        return in_files[0]

    p = Path(filename_to_list(in_files)[0])
    try:
        subj = re.search(r'(?<=^sub-)[a-zA-Z0-9]*', p.name).group()
        suffix = re.search(r'(?<=_)\w+(?=\.)', p.name).group()
    except AttributeError as e:
        raise AttributeError('Could not extract BIDS information') from e

    prefix = f'sub-{subj}'

    if sessionwise:
        ses = re.search(r'(?<=_ses-)[a-zA-Z0-9]*', p.name)
        if ses:
            prefix += f'_ses-{ses.group()}'
    return str(p.parent / f'{prefix}_{suffix}.nii.gz')


def _create_fs_id(subject_id, session_id=None):
    """
    Create FreeSurfer subject ID.

    Examples
    --------
    >>> _create_fs_id('01')
    'sub-01'
    >>> _create_fs_id('sub-01')
    'sub-01'
    >>> _create_fs_id('01', 'pre')
    'sub-01_ses-pre'
    """

    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'

    if session_id:
        ses_str = session_id
        if isinstance(session_id, list):
            from smriprep.utils.misc import stringify_sessions

            ses_str = stringify_sessions(session_id)
        if not ses_str.startswith('ses-'):
            ses_str = f'ses-{ses_str}'
        subject_id += f'_{ses_str}'
    return subject_id
