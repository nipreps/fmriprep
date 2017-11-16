# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from logging import WARNING


class DuplicateLevelFilter(object):
    """
    Keeps track of duplicated warnings
    """

    def __init__(self, level=WARNING):
        self._msgs = {}
        self._level = level

    def filter(self, record):
        """The filter body"""
        if record.levelno < self._level:
            return True

        rvnum = self._msgs.get(record.msg, 0)
        self._msgs[record.msg] = rvnum + 1
        return rvnum == 0

    def get_summary(self, indentation=8):
        """Return the number of times a message was captured"""
        from textwrap import indent
        body = ['* [%d times] %s' % (ntimes, msg)
                for msg, ntimes in self._msgs.items()]
        return indent('\n'.join(body), ' ' * indentation)
