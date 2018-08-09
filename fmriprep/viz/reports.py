#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
fMRIprep reports builder
^^^^^^^^^^^^^^^^^^^^^^^^


"""
from pathlib import Path
import json
import re

import html

import jinja2
from nipype.utils.filemanip import loadcrash, copyfile
from pkg_resources import resource_filename as pkgrf


class Element(object):
    """
    Just a basic component of a report
    """
    def __init__(self, name, title=None):
        self.name = name
        self.title = title


class Reportlet(Element):
    """
    A reportlet has title, description and a list of graphical components
    """
    def __init__(self, name, file_pattern=None, title=None, description=None, raw=False):
        self.name = name
        self.file_pattern = re.compile(file_pattern)
        self.title = title
        self.description = description
        self.source_files = []
        self.contents = []
        self.raw = raw


class SubReport(Element):
    """
    SubReports are sections within a Report
    """
    def __init__(self, name, reportlets=None, title=''):
        self.name = name
        self.title = title
        self.reportlets = []
        if reportlets:
            self.reportlets += reportlets
        self.isnested = False


class Report(object):
    """
    The full report object
    """
    def __init__(self, path, config, out_dir, run_uuid, out_filename='report.html'):
        self.root = path
        self.sections = []
        self.errors = []
        self.out_dir = Path(out_dir)
        self.out_filename = out_filename
        self.run_uuid = run_uuid

        self._load_config(config)

    def _load_config(self, config):
        with open(config, 'r') as configfh:
            config = json.load(configfh)

        self.index(config['sections'])

    def index(self, config):
        fig_dir = 'figures'
        subject_dir = self.root.split('/')[-1]
        subject = re.search('^(?P<subject_id>sub-[a-zA-Z0-9]+)$', subject_dir).group()
        svg_dir = self.out_dir / 'fmriprep' / subject / fig_dir
        svg_dir.mkdir(parents=True, exist_ok=True)
        reportlet_list = list(sorted([str(f) for f in Path(self.root).glob('**/*.*')]))

        for subrep_cfg in config:
            reportlets = []
            for reportlet_cfg in subrep_cfg['reportlets']:
                rlet = Reportlet(**reportlet_cfg)
                for src in reportlet_list:
                    ext = src.split('.')[-1]
                    if rlet.file_pattern.search(src):
                        contents = None
                        if ext == 'html':
                            with open(src) as fp:
                                contents = fp.read().strip()
                        elif ext == 'svg':
                            fbase = Path(src).name
                            copyfile(src, str(svg_dir / fbase),
                                     copy=True, use_hardlink=True)
                            contents = str(Path(subject) / fig_dir / fbase)

                        if contents:
                            rlet.source_files.append(src)
                            rlet.contents.append(contents)

                if rlet.source_files:
                    reportlets.append(rlet)

            if reportlets:
                sub_report = SubReport(
                    subrep_cfg['name'], reportlets=reportlets,
                    title=subrep_cfg.get('title'))
                self.sections.append(order_by_run(sub_report))

        error_dir = self.out_dir / "fmriprep" / subject / 'log' / self.run_uuid
        if error_dir.is_dir():
            self.index_error_dir(error_dir)

    def index_error_dir(self, error_dir):
        """
        Crawl subjects crash directory for the corresponding run and return text for
        .pklz crash file found.
        """
        for crashfile in error_dir.glob('crash*.*'):
            if crashfile.suffix == '.pklz':
                self.errors.append(self._read_pkl(crashfile))
            elif crashfile.suffix == '.txt':
                self.errors.append(self._read_txt(crashfile))

    @staticmethod
    def _read_pkl(path):
        fname = str(path)
        crash_data = loadcrash(fname)
        data = {'file': fname,
                'traceback': ''.join(crash_data['traceback']).replace("\\n", "<br />")}
        if 'node' in crash_data:
            data['node'] = crash_data['node']
            if data['node'].base_dir:
                data['node_dir'] = data['node'].output_dir()
            else:
                data['node_dir'] = "Node crashed before execution"
            data['inputs'] = sorted(data['node'].inputs.trait_get().items())
        return data

    @staticmethod
    def _read_txt(path):
        lines = path.read_text().splitlines()
        data = {'file': str(path)}
        traceback_start = 0
        if lines[0].startswith('Node'):
            data['node'] = lines[0].split(': ', 1)[1]
            data['node_dir'] = lines[1].split(': ', 1)[1]
            inputs = []
            for i, line in enumerate(lines[5:], 5):
                if not line:
                    traceback_start = i + 1
                    break
                inputs.append(tuple(map(html.escape, line.split(' = ', 1))))
            data['inputs'] = sorted(inputs)
        else:
            data['node_dir'] = "Node crashed before execution"
        data['traceback'] = '\n'.join(lines[traceback_start:])
        return data

    def generate_report(self):
        logs_path = self.out_dir / 'fmriprep' / 'logs'

        boilerplate = []
        boiler_idx = 0

        if (logs_path / 'CITATION.html').exists():
            text = (logs_path / 'CITATION.html').read_text()
            text = '<div class="boiler-html">%s</div>' % re.compile(
                '<body>(.*?)</body>',
                re.DOTALL | re.IGNORECASE).findall(text)[0].strip()
            boilerplate.append((boiler_idx, 'HTML', text))
            boiler_idx += 1

        if (logs_path / 'CITATION.md').exists():
            text = '<pre>%s</pre>\n' % (logs_path / 'CITATION.md').read_text()
            boilerplate.append((boiler_idx, 'Markdown', text))
            boiler_idx += 1

        if (logs_path / 'CITATION.tex').exists():
            text = (logs_path / 'CITATION.tex').read_text()
            text = re.compile(
                r'\\begin{document}(.*?)\\end{document}',
                re.DOTALL | re.IGNORECASE).findall(text)[0].strip()
            text = '<pre>%s</pre>\n' % text
            text += '<h3>Bibliography</h3>\n'
            text += '<pre>%s</pre>\n' % Path(
                pkgrf('fmriprep', 'data/boilerplate.bib')).read_text()
            boilerplate.append((boiler_idx, 'LaTeX', text))
            boiler_idx += 1

        searchpath = pkgrf('fmriprep', '/')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=searchpath),
            trim_blocks=True, lstrip_blocks=True
        )
        report_tpl = env.get_template('viz/report.tpl')
        report_render = report_tpl.render(sections=self.sections, errors=self.errors,
                                          boilerplate=boilerplate)

        # Write out report
        (self.out_dir / 'fmriprep' / self.out_filename).write_text(report_render)
        return len(self.errors)


def order_by_run(subreport):
    ordered = []
    run_reps = {}

    for element in subreport.reportlets:
        if len(element.source_files) == 1 and element.source_files[0]:
            ordered.append(element)
            continue

        for filename, file_contents in zip(element.source_files, element.contents):
            name, title = generate_name_title(filename)
            if not filename or not name:
                continue

            new_element = Reportlet(
                name=element.name, title=element.title, file_pattern=element.file_pattern,
                description=element.description, raw=element.raw)
            new_element.contents.append(file_contents)
            new_element.source_files.append(filename)

            if name not in run_reps:
                run_reps[name] = SubReport(name, title=title)

            run_reps[name].reportlets.append(new_element)

    if run_reps:
        keys = list(sorted(run_reps.keys()))
        for key in keys:
            ordered.append(run_reps[key])
        subreport.isnested = True

    subreport.reportlets = ordered
    return subreport


def generate_name_title(filename):
    fname = Path(filename).name
    expr = re.compile('^sub-(?P<subject_id>[a-zA-Z0-9]+)(_ses-(?P<session_id>[a-zA-Z0-9]+))?'
                      '(_task-(?P<task_id>[a-zA-Z0-9]+))?(_acq-(?P<acq_id>[a-zA-Z0-9]+))?'
                      '(_rec-(?P<rec_id>[a-zA-Z0-9]+))?(_run-(?P<run_id>[a-zA-Z0-9]+))?')
    outputs = expr.search(fname)
    if outputs:
        outputs = outputs.groupdict()
    else:
        return None, None

    name = '{session}{task}{acq}{rec}{run}'.format(
        session="_ses-" + outputs['session_id'] if outputs['session_id'] else '',
        task="_task-" + outputs['task_id'] if outputs['task_id'] else '',
        acq="_acq-" + outputs['acq_id'] if outputs['acq_id'] else '',
        rec="_rec-" + outputs['rec_id'] if outputs['rec_id'] else '',
        run="_run-" + outputs['run_id'] if outputs['run_id'] else ''
    )
    title = '{session}{task}{acq}{rec}{run}'.format(
        session=" Session: " + outputs['session_id'] if outputs['session_id'] else '',
        task=" Task: " + outputs['task_id'] if outputs['task_id'] else '',
        acq=" Acquisition: " + outputs['acq_id'] if outputs['acq_id'] else '',
        rec=" Reconstruction: " + outputs['rec_id'] if outputs['rec_id'] else '',
        run=" Run: " + outputs['run_id'] if outputs['run_id'] else ''
    )
    return name.strip('_'), title


def run_reports(reportlets_dir, out_dir, subject_label, run_uuid):
    """
    Runs the reports

    >>> import os
    >>> from shutil import copytree
    >>> from tempfile import TemporaryDirectory
    >>> filepath = os.path.dirname(os.path.realpath(__file__))
    >>> test_data_path = os.path.realpath(os.path.join(filepath,
    ...                                   '../data/tests/work'))
    >>> curdir = os.getcwd()
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    >>> data_dir = copytree(test_data_path, os.path.abspath('work'))
    >>> os.makedirs('out/fmriprep', exist_ok=True)
    >>> run_reports(os.path.abspath('work/reportlets'),
    ...             os.path.abspath('out'),
    ...             '01', 'madeoutuuid')
    0
    >>> os.chdir(curdir)
    >>> tmpdir.cleanup()

    """
    reportlet_path = str(Path(reportlets_dir) / 'fmriprep' / ("sub-%s" % subject_label))
    config = pkgrf('fmriprep', 'viz/config.json')

    out_filename = 'sub-{}.html'.format(subject_label)
    report = Report(reportlet_path, config, out_dir, run_uuid, out_filename)
    return report.generate_report()


def generate_reports(subject_list, output_dir, work_dir, run_uuid):
    """
    A wrapper to run_reports on a given ``subject_list``
    """
    reports_dir = str(Path(work_dir) / 'reportlets')
    report_errors = [
        run_reports(reports_dir, output_dir, subject_label, run_uuid=run_uuid)
        for subject_label in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging
        logger = logging.getLogger('cli')
        logger.warning(
            'Errors occurred while generating reports for participants: %s.',
            ', '.join(['%s (%d)' % (subid, err)
                       for subid, err in zip(subject_list, report_errors)]))
    return errno
