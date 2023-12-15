# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
from pathlib import Path

from nireports.assembler.report import Report

MAX_SES_AGR = 4
"""Maximum number of sessions aggregated in one subject's visual report. If exceeded, visual reports are separated per session."""


def generate_reports(subject_list, output_dir, run_uuid, config=None, work_dir=None):
    """Generate reports for a list of subjects."""
    from .. import config, data

    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"

    report_errors = []
    for subject_label in subject_list:
        entities = {}
        entities["subject"] = subject_label

        session_list = config.execution.layout.get_sessions(subject=subject_label)

        if config is not None:
            # If a config file is precised, we do not override it
            html_report = "report.html"
        elif len(session_list) < MAX_SES_AGR:
            # If there is only a few session for this subject, we aggregate them in a single visual report.
            config = data.load("reports-spec.yml")
            html_report = "report.html"
        else:
            # Beyond a threshold, we separate the anatomical report from the functional.
            config = data.load("reports-spec-anat.yml")
            html_report = ''.join([f"sub-{subject_label}", "_anat.html"])

        robj = Report(
            output_dir,
            run_uuid,
            bootstrap_file=config,
            out_filename=html_report,
            reportlets_dir=reportlets_dir,
            plugins=None,
            plugin_meta=None,
            metadata=None,
            **entities,
        )

        # Count nbr of subject for which report generation failed
        errno = 0
        try:
            robj.generate_report()
        except:
            errno += 1

        if len(session_list) >= MAX_SES_AGR:
            # Beyond a certain number of sessions per subject, we separate the functional reports per session
            for session_label in session_list:
                config = data.load("reports-spec-func.yml")
                html_report = ''.join(
                    [f"sub-{subject_label}", f"_ses-{session_label}", "_func.html"]
                )
                entities["session"] = session_label

                robj = Report(
                    output_dir,
                    run_uuid,
                    bootstrap_file=config,
                    out_filename=html_report,
                    reportlets_dir=reportlets_dir,
                    plugins=None,
                    plugin_meta=None,
                    metadata=None,
                    **entities,
                )

            # Add up the nbr of subject for which report generation failed
            try:
                robj.generate_report()
            except:
                errno += 1

    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            f"{subid} ({err})" for subid, err in zip(subject_list, report_errors) if err
        )
        logger.error(
            "Preprocessing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    return errno
