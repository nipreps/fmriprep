from __future__ import unicode_literals

import json
import re
import os
import time

import jinja2
from niworkflows.nipype.utils.filemanip import loadcrash
from pkg_resources import resource_filename as pkgrf

from .. import __version__


class Element(object):
    def __init__(self, name, file_pattern, title=None, description=None, raw=False):
        self.name = name
        self.file_pattern = re.compile(file_pattern)
        self.title = title
        self.description = description
        self.files_contents = []
        self.raw = raw


class SubReport(object):
    def __init__(self, name, elements, title=''):
        self.name = name
        self.title = title
        self.run_reports = []
        self.elements = [Element(**e) for e in elements]

    def order_by_run(self):
        run_reps = {}
        for element in self.elements:
            for index in range(len(element.files_contents) - 1, -1, -1):
                filename = element.files_contents[index][0]
                file_contents = element.files_contents[index][1]
                name, title = self.generate_name_title(filename)
                if not name:
                    continue
                new_elem = {'name': element.name, 'file_pattern': element.file_pattern,
                            'title': element.title, 'description': element.description,
                            'raw': element.raw}
                try:
                    new_element = Element(**new_elem)
                    run_reps[name].elements.append(new_element)
                    run_reps[name].elements[-1].files_contents.append((filename, file_contents))
                except KeyError:
                    run_reps[name] = SubReport(name, [new_elem], title=title)
                    run_reps[name].elements[0].files_contents.append((filename, file_contents))
        keys = list(run_reps.keys())
        keys.sort()
        for key in keys:
            self.run_reports.append(run_reps[key])

    def generate_name_title(self, filename):
        fname = os.path.basename(filename)
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
        return name, title


class Report(object):

    def __init__(self, path, config, out_dir, run_uuid, out_filename='report.html'):
        self.root = path
        self.sub_reports = []
        self.errors = []
        self.out_dir = out_dir
        self.out_filename = out_filename
        self.run_uuid = run_uuid

        self._load_config(config)

    def _load_config(self, config):
        config = json.load(open(config, 'r'))

        for e in config['sub_reports']:
            sub_report = SubReport(**e)
            self.sub_reports.append(sub_report)

        self.index()

    def index(self):
        for root, directories, filenames in os.walk(self.root):
            for f in filenames:
                f = os.path.join(root, f)
                for sub_report in self.sub_reports:
                    for element in sub_report.elements:
                        ext = f.split('.')[-1]
                        if element.file_pattern.search(f) and (ext == 'svg' or ext == 'html'):
                            with open(f) as fp:
                                content = fp.read()
                                if not element.raw:
                                    content = content.split('\n', 1)[1]
                                element.files_contents.append((f, content))
        for sub_report in self.sub_reports:
            sub_report.order_by_run()

        subject_dir = self.root.split('/')[-1]
        subject = re.search('^(?P<subject_id>sub-[a-zA-Z0-9]+)$', subject_dir).group()
        error_dir = os.path.join(self.out_dir, "fmriprep", subject, 'log', self.run_uuid)
        if os.path.isdir(error_dir):
            self.index_error_dir(error_dir)

    def index_error_dir(self, error_dir):
        ''' Crawl subjects crash directory for the corresponding run and return text for
            .pklz crash file found. '''
        for root, directories, filenames in os.walk(error_dir):
            for f in filenames:
                # Only deal with files that start with crash and end in pklz
                if not (f[:5] == 'crash' and f[-4:] == 'pklz'):
                    continue
                crash_data = loadcrash(os.path.join(root, f))
                error = {}
                node = None
                if 'node' in crash_data:
                    node = crash_data['node']
                error['traceback'] = ''.join(crash_data['traceback']).replace("\\n", "<br \>")
                error['file'] = f

                if node:
                    error['node'] = node
                    if node.base_dir:
                        error['node_dir'] = node.output_dir()
                    else:
                        error['node_dir'] = "Node crashed before execution"
                    error['inputs'] = sorted(node.inputs.trait_get().items())
                self.errors.append(error)

    def generate_report(self):
        searchpath = pkgrf('fmriprep', '/')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=searchpath),
            trim_blocks=True, lstrip_blocks=True
        )
        report_tpl = env.get_template('viz/report.tpl')
        # Ignore subreports with no children
        sub_reports = [sub_report for sub_report in self.sub_reports
                       if len(sub_report.run_reports) > 0 or
                       any(elem.files_contents for elem in sub_report.elements)]
        report_render = report_tpl.render(sub_reports=sub_reports, errors=self.errors,
                                          date=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                          version=__version__)
        with open(os.path.join(self.out_dir, "fmriprep", self.out_filename), 'w') as fp:
            fp.write(report_render)
        return len(self.errors)


def run_reports(reportlets_dir, out_dir, subject_label, run_uuid):
    reportlet_path = os.path.join(reportlets_dir, 'fmriprep', "sub-" + subject_label)
    config = pkgrf('fmriprep', 'viz/config.json')

    out_filename = 'sub-{}.html'.format(subject_label)
    report = Report(reportlet_path, config, out_dir, run_uuid, out_filename)
    return report.generate_report()
