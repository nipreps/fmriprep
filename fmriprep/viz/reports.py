from __future__ import unicode_literals

import json
import re
import os

import jinja2
from pkg_resources import resource_filename as pkgrf


class Element(object):

    def __init__(self, name, file_pattern, title, description):
        self.name = name
        self.file_pattern = re.compile(file_pattern)
        self.title = title
        self.description = description
        self.files = []
        self.files_contents = []
        self.files_subjects = []


class SubReport(object):

    def __init__(self, name, elements, **kwargs):
        self.name = name
        self.elements = []
        for e in elements:
            element = Element(**e)
            self.elements.append(element)

    def generate_sub_report(self, report):
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath='/'),
            trim_blocks=True, lstrip_blocks=True
        )
        sub_report_tpl = env.get_template('{}_tpl.html'.format(self.name))
        sub_report_render = sub_report_tpl.render
        return sub_report_render


class Report(object):

    def __init__(self, path, config, out_dir, out_filename='report.html'):
        self.root = path
        self.sub_reports = []
        self._load_config(config)
        self.out_dir = out_dir
        self.out_filename = out_filename

    def _load_config(self, config):
        try:
            config = json.load(open(config, 'r'))
        except Exception as e:
            print(e)
            return

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
                        if element.file_pattern.search(f) and f.split('.')[-1] == 'svg':
                            element.files.append(f)
                            with open(f) as fp:
                                content = fp.read()
                                content = '\n'.join(content.split('\n')[1:])
                                element.files_contents.append((f, content))

    def generate_report(self):
        searchpath = pkgrf('fmriprep', '/')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=searchpath),
            trim_blocks=True, lstrip_blocks=True
        )
        report_tpl = env.get_template('viz/report.tpl')
        report_render = report_tpl.render(sub_reports=self.sub_reports)
        with open(os.path.join(self.out_dir, self.out_filename), 'w') as fp:
            fp.write(report_render)
        return report_render


def run_reports(out_dir):
    path = os.path.join(out_dir, 'images/')
    config = pkgrf('fmriprep', 'viz/config.json')

    for root, _, _ in os.walk(path):
        #  relies on the fact that os.walk does not return a trailing /
        dir = root.split('/')[-1]
        try:
            subject = re.search('^(?P<subject_id>sub-[a-zA-Z0-9]+)$', dir).group()
            out_filename = '{}{}'.format(subject, '.html')
            report = Report(path, config, out_dir, out_filename)
            report.generate_report()
        except AttributeError:
            continue
