# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Supercharging Nipype's workflow engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add special features to the Nipype's vanilla workflows
"""
import networkx as nx
from nipype.pipeline import engine as pe


class LiterateWorkflow(pe.Workflow):
    """Controls the setup and execution of a pipeline of processes."""

    def __init__(self, name, base_dir=None):
        """Create a workflow object.
        Parameters
        ----------
        name : alphanumeric string
            unique identifier for the workflow
        base_dir : string, optional
            path to workflow storage
        """
        super(LiterateWorkflow, self).__init__(name, base_dir)
        self.__desc__ = None
        self.__postdesc__ = None

    def visit_desc(self):
        """
        Builds a citation boilerplate by visiting all workflows
        appending their ``__desc__`` field
        """
        desc = []

        if self.__desc__:
            desc += [self.__desc__]

        for node in nx.topological_sort(self._graph):
            if isinstance(node, LiterateWorkflow):
                add_desc = node.visit_desc()
                if add_desc not in desc:
                    desc.append(add_desc)

        if self.__postdesc__:
            desc += [self.__postdesc__]

        return ''.join(desc)
