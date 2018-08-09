# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling confounds
^^^^^^^^^^^^^^^^^^

    >>> import os
    >>> import pandas as pd

"""
import os
import shutil
import numpy as np
import pandas as pd
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)
from niworkflows.viz.plots import fMRIPlot

LOGGER = logging.getLogger('nipype.interface')


class GatherConfoundsInputSpec(BaseInterfaceInputSpec):
    signals = File(exists=True, desc='input signals')
    dvars = File(exists=True, desc='file containing DVARS')
    fd = File(exists=True, desc='input framewise displacement')
    tcompcor = File(exists=True, desc='input tCompCorr')
    acompcor = File(exists=True, desc='input aCompCorr')
    cos_basis = File(exists=True, desc='input cosine basis')
    motion = File(exists=True, desc='input motion parameters')
    aroma = File(exists=True, desc='input ICA-AROMA')


class GatherConfoundsOutputSpec(TraitedSpec):
    confounds_file = File(exists=True, desc='output confounds file')
    confounds_list = traits.List(traits.Str, desc='list of headers')


class GatherConfounds(SimpleInterface):
    """
    Combine various sources of confounds in one TSV file

    .. testsetup::

    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)

    .. doctest::

    >>> pd.DataFrame({'a': [0.1]}).to_csv('signals.tsv', index=False, na_rep='n/a')
    >>> pd.DataFrame({'b': [0.2]}).to_csv('dvars.tsv', index=False, na_rep='n/a')

    >>> gather = GatherConfounds()
    >>> gather.inputs.signals = 'signals.tsv'
    >>> gather.inputs.dvars = 'dvars.tsv'
    >>> res = gather.run()
    >>> res.outputs.confounds_list
    ['Global signals', 'DVARS']

    >>> pd.read_csv(res.outputs.confounds_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
         a    b
    0  0.1  0.2

    .. testcleanup::

    >>> tmpdir.cleanup()

    """
    input_spec = GatherConfoundsInputSpec
    output_spec = GatherConfoundsOutputSpec

    def _run_interface(self, runtime):
        combined_out, confounds_list = _gather_confounds(
            signals=self.inputs.signals,
            dvars=self.inputs.dvars,
            fdisp=self.inputs.fd,
            tcompcor=self.inputs.tcompcor,
            acompcor=self.inputs.acompcor,
            cos_basis=self.inputs.cos_basis,
            motion=self.inputs.motion,
            aroma=self.inputs.aroma,
            newpath=runtime.cwd,
        )
        self._results['confounds_file'] = combined_out
        self._results['confounds_list'] = confounds_list
        return runtime


class ICAConfoundsInputSpec(BaseInterfaceInputSpec):
    in_directory = Directory(mandatory=True, desc='directory where ICA derivatives are found')
    ignore_aroma_err = traits.Bool(False, usedefault=True, desc='ignore ICA-AROMA errors')


class ICAConfoundsOutputSpec(TraitedSpec):
    aroma_confounds = File(exists=True, desc='output confounds file extracted from ICA-AROMA')
    aroma_noise_ics = File(exists=True, desc='ICA-AROMA noise components')
    melodic_mix = File(exists=True, desc='melodic mix file')


class ICAConfounds(SimpleInterface):
    """Extract confounds from ICA-AROMA result directory
    """
    input_spec = ICAConfoundsInputSpec
    output_spec = ICAConfoundsOutputSpec

    def _run_interface(self, runtime):
        aroma_confounds, motion_ics_out, melodic_mix_out = _get_ica_confounds(
            self.inputs.in_directory, newpath=runtime.cwd)

        if aroma_confounds is not None:
            self._results['aroma_confounds'] = aroma_confounds
        elif not self.inputs.ignore_aroma_err:
            raise RuntimeError('ICA-AROMA failed')

        self._results['aroma_noise_ics'] = motion_ics_out
        self._results['melodic_mix'] = melodic_mix_out
        return runtime


def _gather_confounds(signals=None, dvars=None, fdisp=None,
                      tcompcor=None, acompcor=None, cos_basis=None,
                      motion=None, aroma=None, newpath=None):
    """
    Load confounds from the filenames, concatenate together horizontally
    and save new file.

    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    >>> pd.DataFrame({'a': [0.1]}).to_csv('signals.tsv', index=False, na_rep='n/a')
    >>> pd.DataFrame({'b': [0.2]}).to_csv('dvars.tsv', index=False, na_rep='n/a')
    >>> out_file, confound_list = _gather_confounds('signals.tsv', 'dvars.tsv')
    >>> confound_list
    ['Global signals', 'DVARS']

    >>> pd.read_csv(out_file, sep='\s+', index_col=None,
    ...             engine='python')  # doctest: +NORMALIZE_WHITESPACE
         a    b
    0  0.1  0.2
    >>> tmpdir.cleanup()


    """

    def less_breakable(a_string):
        ''' hardens the string to different envs (i.e. case insensitive, no whitespace, '#' '''
        return ''.join(a_string.split()).strip('#')

    def _adjust_indices(left_df, right_df):
        # This forces missing values to appear at the beggining of the DataFrame
        # instead of the end
        index_diff = len(left_df.index) - len(right_df.index)
        if index_diff > 0:
            right_df.index = range(index_diff,
                                   len(right_df.index) + index_diff)
        elif index_diff < 0:
            left_df.index = range(-index_diff,
                                  len(left_df.index) - index_diff)

    all_files = []
    confounds_list = []
    for confound, name in ((signals, 'Global signals'),
                           (dvars, 'DVARS'),
                           (fdisp, 'Framewise displacement'),
                           (tcompcor, 'tCompCor'),
                           (acompcor, 'aCompCor'),
                           (cos_basis, 'Cosine basis'),
                           (motion, 'Motion parameters'),
                           (aroma, 'ICA-AROMA')):
        if confound is not None and isdefined(confound):
            confounds_list.append(name)
            if os.path.exists(confound) and os.stat(confound).st_size > 0:
                all_files.append(confound)

    confounds_data = pd.DataFrame()
    for file_name in all_files:  # assumes they all have headings already
        new = pd.read_csv(file_name, sep="\t")
        for column_name in new.columns:
            new.rename(columns={column_name: less_breakable(column_name)},
                       inplace=True)

        _adjust_indices(confounds_data, new)
        confounds_data = pd.concat((confounds_data, new), axis=1)

    if newpath is None:
        newpath = os.getcwd()

    combined_out = os.path.join(newpath, 'confounds.tsv')
    confounds_data.to_csv(combined_out, sep='\t', index=False,
                          na_rep='n/a')

    return combined_out, confounds_list


def _get_ica_confounds(ica_out_dir, newpath=None):
    if newpath is None:
        newpath = os.getcwd()

    # load the txt files from ICA-AROMA
    melodic_mix = os.path.join(ica_out_dir, 'melodic.ica/melodic_mix')
    motion_ics = os.path.join(ica_out_dir, 'classified_motion_ICs.txt')

    # Change names of motion_ics and melodic_mix for output
    melodic_mix_out = os.path.join(newpath, 'MELODICmix.tsv')
    motion_ics_out = os.path.join(newpath, 'AROMAnoiseICs.csv')

    # melodic_mix replace spaces with tabs
    with open(melodic_mix, 'r') as melodic_file:
        melodic_mix_out_char = melodic_file.read().replace('  ', '\t')
    # write to output file
    with open(melodic_mix_out, 'w+') as melodic_file_out:
        melodic_file_out.write(melodic_mix_out_char)

    # copy metion_ics file to derivatives name
    shutil.copyfile(motion_ics, motion_ics_out)

    # -1 since python lists start at index 0
    motion_ic_indices = np.loadtxt(motion_ics, dtype=int, delimiter=',', ndmin=1) - 1
    melodic_mix_arr = np.loadtxt(melodic_mix, ndmin=2)

    # Return dummy list of ones if no noise compnents were found
    if motion_ic_indices.size == 0:
        LOGGER.warning('No noise components were classified')
        return None, motion_ics_out, melodic_mix_out

    # the "good" ics, (e.g. not motion related)
    good_ic_arr = np.delete(melodic_mix_arr, motion_ic_indices, 1).T

    # return dummy lists of zeros if no signal components were found
    if good_ic_arr.size == 0:
        LOGGER.warning('No signal components were classified')
        return None, motion_ics_out, melodic_mix_out

    # transpose melodic_mix_arr so x refers to the correct dimension
    aggr_confounds = np.asarray([melodic_mix_arr.T[x] for x in motion_ic_indices])

    # add one to motion_ic_indices to match melodic report.
    aroma_confounds = os.path.join(newpath, "AROMAAggrCompAROMAConfounds.tsv")
    pd.DataFrame(aggr_confounds.T,
                 columns=['AROMAAggrComp%02d' % (x + 1) for x in motion_ic_indices]).to_csv(
        aroma_confounds, sep="\t", index=None)

    return aroma_confounds, motion_ics_out, melodic_mix_out


class FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_func = File(exists=True, mandatory=True,
                   desc='input BOLD time-series (4D file)')
    in_mask = File(exists=True, mandatory=True,
                   desc='3D brain mask')
    in_segm = File(exists=True, desc='resampled segmentation')
    confounds_file = File(exists=True,
                          desc="BIDS' _confounds.tsv file")

    str_or_tuple = traits.Either(
        traits.Str,
        traits.Tuple(traits.Str, traits.Either(None, traits.Str)),
        traits.Tuple(traits.Str, traits.Either(None, traits.Str), traits.Either(None, traits.Str)))
    confounds_list = traits.List(
        str_or_tuple, minlen=1,
        desc='list of headers to extract from the confounds_file')
    tr = traits.Either(None, traits.Float, usedefault=True,
                       desc='the repetition time')


class FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class FMRISummary(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """
    input_spec = FMRISummaryInputSpec
    output_spec = FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_func,
            suffix='_fmriplot.svg',
            use_ext=False,
            newpath=runtime.cwd)

        dataframe = pd.read_csv(
            self.inputs.confounds_file,
            sep="\t", index_col=None, dtype='float32',
            na_filter=True, na_values='n/a')

        headers = []
        units = {}
        names = {}

        for conf_el in self.inputs.confounds_list:
            if isinstance(conf_el, (list, tuple)):
                headers.append(conf_el[0])
                if conf_el[1] is not None:
                    units[conf_el[0]] = conf_el[1]

                if len(conf_el) > 2 and conf_el[2] is not None:
                    names[conf_el[0]] = conf_el[2]
            else:
                headers.append(conf_el)

        if not headers:
            data = None
            units = None
        else:
            data = dataframe[headers]

        colnames = data.columns.ravel().tolist()

        for name, newname in list(names.items()):
            colnames[colnames.index(name)] = newname

        data.columns = colnames

        fig = fMRIPlot(
            self.inputs.in_func,
            mask_file=self.inputs.in_mask,
            seg_file=(self.inputs.in_segm
                      if isdefined(self.inputs.in_segm) else None),
            tr=self.inputs.tr,
            data=data,
            units=units,
        ).plot()
        fig.savefig(self._results['out_file'], bbox_inches='tight')
        return runtime
