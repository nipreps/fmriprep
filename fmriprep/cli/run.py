# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""fMRI preprocessing workflow."""

from .. import config


def main():
    """Entry point."""
    import gc
    import sys
    from multiprocessing import Manager, Process
    from os import EX_SOFTWARE
    from pathlib import Path

    from ..utils.bids import write_bidsignore, write_derivative_description
    from .parser import parse_args
    from .workflow import build_workflow

    parse_args()

    # Deprecated flags
    if any(
        (
            config.workflow.use_aroma,
            config.workflow.aroma_err_on_warn,
            config.workflow.aroma_melodic_dim,
        )
    ):
        config.loggers.cli.warning(
            'ICA-AROMA was removed in fMRIPrep 23.1.0. The --use-aroma, --aroma-err-on-warn, '
            'and --aroma-melodic-dim flags will error in fMRIPrep 24.0.0.'
        )

    # Code Carbon
    if config.execution.track_carbon:
        from codecarbon import OfflineEmissionsTracker

        country_iso_code = config.execution.country_code
        config.loggers.workflow.log(25, 'CodeCarbon tracker started ...')
        config.loggers.workflow.log(25, f'Using country_iso_code: {country_iso_code}')
        config.loggers.workflow.log(25, f'Saving logs at: {config.execution.log_dir}')

        tracker = OfflineEmissionsTracker(
            output_dir=config.execution.log_dir, country_iso_code=country_iso_code
        )
        tracker.start()

    if 'pdb' in config.execution.debug:
        from fmriprep.utils.debug import setup_exceptionhook

        setup_exceptionhook()
        config.nipype.plugin = 'Linear'

    sentry_sdk = None
    if not config.execution.notrack and not config.execution.debug:
        import sentry_sdk

        from ..utils.telemetry import sentry_setup, setup_migas

        sentry_setup()
        setup_migas(init_ping=True)

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    config_file = config.execution.work_dir / config.execution.run_uuid / 'config.toml'
    config_file.parent.mkdir(exist_ok=True, parents=True)
    config.to_filename(config_file)

    # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
    # Because Python on Linux does not ever free virtual memory (VM), running the
    # workflow construction jailed within a process preempts excessive VM buildup.
    if 'pdb' not in config.execution.debug:
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), retval))
            p.start()
            p.join()
            retval = dict(retval.items())  # Convert to base dictionary

            if p.exitcode:
                retval['return_code'] = p.exitcode

    else:
        retval = build_workflow(str(config_file), {})

    exitcode = retval.get('return_code', 0)
    fmriprep_wf = retval.get('workflow', None)

    # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
    # function executed constrained in a process may change the config (and thus the global
    # state of fMRIPrep).
    config.load(config_file)

    if config.execution.reports_only:
        sys.exit(int(exitcode > 0))

    if fmriprep_wf and config.execution.write_graph:
        fmriprep_wf.write_graph(graph2use='colored', format='svg', simple_form=True)

    exitcode = exitcode or (fmriprep_wf is None) * EX_SOFTWARE
    if exitcode != 0:
        sys.exit(exitcode)

    # Generate boilerplate
    with Manager() as mgr:
        from .workflow import build_boilerplate

        p = Process(target=build_boilerplate, args=(str(config_file), fmriprep_wf))
        p.start()
        p.join()

    if config.execution.boilerplate_only:
        sys.exit(int(exitcode > 0))

    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Sentry tracking
    if sentry_sdk is not None:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag('run_uuid', config.execution.run_uuid)
            scope.set_tag('npart', len(config.execution.participant_label))
        sentry_sdk.add_breadcrumb(message='fMRIPrep started', level='info')
        sentry_sdk.capture_message('fMRIPrep started', level='info')

    config.loggers.workflow.log(
        15,
        '\n'.join(['fMRIPrep config:'] + [f'\t\t{s}' for s in config.dumps().splitlines()]),
    )
    config.loggers.workflow.log(25, 'fMRIPrep started!')
    errno = 1  # Default is error exit unless otherwise set
    try:
        fmriprep_wf.run(**config.nipype.get_plugin())
    except Exception as e:
        if not config.execution.notrack:
            from ..utils.telemetry import process_crashfile

            crashfolders = [
                config.execution.fmriprep_dir / f'sub-{s}' / 'log' / config.execution.run_uuid
                for s in config.execution.participant_label
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob('crash*.*'):
                    process_crashfile(crashfile)

            if sentry_sdk is not None and 'Workflow did not execute cleanly' not in str(e):
                sentry_sdk.capture_exception(e)
        config.loggers.workflow.critical('fMRIPrep failed: %s', e)
        raise
    else:
        config.loggers.workflow.log(25, 'fMRIPrep finished successfully!')
        if sentry_sdk is not None:
            success_message = 'fMRIPrep finished without errors'
            sentry_sdk.add_breadcrumb(message=success_message, level='info')
            sentry_sdk.capture_message(success_message, level='info')

        # Bother users with the boilerplate only iff the workflow went okay.
        boiler_file = config.execution.fmriprep_dir / 'logs' / 'CITATION.md'
        if boiler_file.exists():
            if config.environment.exec_env in (
                'singularity',
                'docker',
                'fmriprep-docker',
            ):
                boiler_file = Path('<OUTPUT_PATH>') / boiler_file.relative_to(
                    config.execution.output_dir
                )
            config.loggers.workflow.log(
                25,
                'Works derived from this fMRIPrep execution should include the '
                f'boilerplate text found in {boiler_file}.',
            )

        if config.workflow.run_reconall:
            from niworkflows.utils.misc import _copy_any
            from templateflow import api

            dseg_tsv = str(
                api.get(
                    'fsaverage',
                    hemi=None,
                    atlas=None,
                    segmentation='aparc',
                    suffix='dseg',
                    extension=['.tsv'],
                )
            )
            _copy_any(dseg_tsv, str(config.execution.fmriprep_dir / 'desc-aseg_dseg.tsv'))
            _copy_any(dseg_tsv, str(config.execution.fmriprep_dir / 'desc-aparcaseg_dseg.tsv'))
        errno = 0
    finally:
        # Code Carbon
        if config.execution.track_carbon:
            emissions: float = tracker.stop()
            config.loggers.workflow.log(25, 'CodeCarbon tracker has stopped.')
            config.loggers.workflow.log(25, f'Saving logs at: {config.execution.log_dir}')
            config.loggers.workflow.log(25, f'Carbon emissions: {emissions} kg')

        from fmriprep.reports.core import generate_reports

        # Generate reports phase
        session_list = (
            config.execution.get().get('bids_filters', {}).get('bold', {}).get('session')
        )

        failed_reports = generate_reports(
            config.execution.participant_label,
            config.execution.fmriprep_dir,
            config.execution.run_uuid,
            session_list=session_list,
        )
        write_derivative_description(
            config.execution.bids_dir,
            config.execution.fmriprep_dir,
            dataset_links=config.execution.dataset_links,
        )
        write_bidsignore(config.execution.fmriprep_dir)

        if failed_reports:
            msg = (
                'Report generation was not successful for the following participants '
                f': {", ".join(failed_reports)}.'
            )
            config.loggers.cli.error(msg)
            if sentry_sdk is not None:
                sentry_sdk.capture_message(msg, level='error')

        sys.exit(int((errno + len(failed_reports)) > 0))
