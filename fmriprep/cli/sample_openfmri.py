# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
A tool to sample OpenfMRI datasets using a datalad installation

Please run this first: ::

    # install openfmri dataset
    datalad install -r ///openfmri

    # get all sidecar files
    datalad get -J8 $( find openfmri/ -name "*_T1w.json" )
    datalad get -J8 $( find openfmri/ -name "*_T2w.json" )
    datalad get -J8 $( find openfmri/ -name "*_bold.json" )
    datalad get -J8 $( find openfmri/ -name "*_magnitude*.json" )
    datalad get -J8 $( find openfmri/ -name "*_phase*.json" )
    datalad get -J8 $( find openfmri/ -name "*_fieldmap.json" )

    # list subjects
    cd openfmri


"""

import os
import glob
import numpy as np


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(
        description='OpenfMRI participants sampler, for FMRIPREP\'s testing purposes',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('openfmri_dir', action='store',
                        help='the root folder of a the openfmri dataset')

    # optional arguments
    parser.add_argument('-D', '--datalad_fetch', action='store_true', default=False,
                        help='download sampled subjects')
    parser.add_argument('-o', '--output-file', action='store', help='write output file')
    parser.add_argument('-n', '--num-participants', action='store', type=int, default=4,
                        help='number of participants randomly selected per dataset')
    parser.add_argument('--njobs', action='store', type=int, help='parallel downloads')
    parser.add_argument('--seed', action='store', type=int, default=20170914,
                        help='seed for random number generation')

    return parser


def main():
    """Entry point"""
    thispath = os.getcwd()
    opts = get_parser().parse_args()
    np.random.seed(opts.seed)

    out_file = None
    if opts.output_file is not None:
        out_file = os.path.abspath(opts.output_file)

    os.chdir(opts.openfmri_dir)
    all_sub = sorted(glob.glob('ds*/sub-*'))
    datasets = {}
    multises = set()
    for subj in all_sub:
        ds = subj.split('/')[0]
        if os.path.isdir(os.path.join(subj, 'anat')) and os.path.isdir(os.path.join(subj, 'func')):
            datasets.setdefault(ds, []).append(os.path.basename(subj))
        elif (glob.glob(os.path.join(subj, 'ses-*', 'anat')) and
              glob.glob(os.path.join(subj, 'ses-*', 'func'))):
            multises.add(ds)
            datasets.setdefault(ds, []).append(os.path.basename(subj))

    subsample = {}

    n_sample = 0
    for ds, sublist in datasets.items():
        n_sample += min(opts.num_participants, len(sublist))
        if len(sublist) <= opts.num_participants:
            subsample[ds] = sublist
        else:
            subsample[ds] = sorted(np.random.choice(
                sublist, size=opts.num_participants, replace=False).tolist())

    # Double check everything looks good
    assert n_sample == len([sub for _, sublist in datasets.items() for sub in sublist])

    if out_file is not None:
        import yaml
        with open(out_file, 'w') as outfh:
            outfh.write(yaml.dump(subsample))
        print('Sampled participants stored to %s' % out_file)

    singleses = set(datasets.keys()) - multises
    print('Sampled %d participants' % n_sample)
    print('Datasets summary:\n\tSingle-session=%d'
          '\n\tMulti-session=%d'
          '\n\tTotal participants=%d' % (len(singleses), len(multises), n_sample))
    os.chdir(thispath)

    if opts.datalad_fetch:
        import datalad.api as dlad
        for ds, sublist in subsample.items():
            for sub in sublist:
                dlad.get(path=os.path.join(opts.openfmri_dir, ds, sub),
                         recursive=True, jobs=opts.njobs, verbose=True)


if __name__ == '__main__':
    main()
