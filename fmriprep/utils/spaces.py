from ..config import NONSTANDARD_REFERENCES

FSAVERAGE_DENSITY = {
    '642': 'fsaverage3',
    '2562': 'fsaverage4',
    '10k': 'fsaverage5',
    '41k': 'fsaverage6',
    '164k': 'fsaverage',
}


class Spaces:
    """Helper class for managing fMRIPrep spaces."""
    __slots__ = ('_output', '_internal')

    def __init__(self, output=None):
        """
        Tracks and distinguishes internal and output spaces.

        Internal spaces are normalizations required for pipeline execution which
        can vary based on user arguments.
        Output spaces are desired user outputs.
        """
        self._output = []
        self._internal = []
        if output is not None:
            self.output = output

    def __repr__(self):
        return 'Output spaces: %s\nInternal spaces: %s' % (self.output, self.internal)

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, vals):
        if not hasattr(vals, '__iter__'):
            return
        for tpl, specs in vals:
            self.add_space(tpl, specs=specs)

    @property
    def internal(self):
        return self._internal

    @property
    def all(self):
        return self.output + self.internal

    def unique(self, spaces='all'):
        """Return unique spaces from an iterable composed of (template, specs) tuples"""
        return {space[0] for space in getattr(self, spaces)}

    def add_space(self, template, specs=None, output=True, strict=True):
        if specs is None or not isinstance(specs, dict):
            specs = {}
        # map densities to fsaverage subject
        if template == 'fsaverage' and specs.get('den') in FSAVERAGE_DENSITY:
            template = FSAVERAGE_DENSITY[specs.get('den')]
            del specs['den']

        attr = self.output if output is True else self.internal
        space = (template, specs)
        if strict and space in self.all:
            # avoid duplication
            return
        attr.append(space)

    def filtered(self, flt, spaces='all', name_only=False):
        methods = {
            'std': _filter_std,
            'surf': _filter_surf,
            'vol': _filter_vol,
            'std_vol': _filter_std_vols,
        }
        if flt not in methods:
            return
        func = methods[flt]
        filtered_spaces = [space for space in getattr(self, spaces) if func(space[0])]
        if name_only:
            filtered_spaces = [space for space, specs in filtered_spaces]
        return filtered_spaces

    def get_space(self, template, spaces='all', specs=None):
        """
        Query ``spaces`` and return the first matching ``template`` entry.
        If no matches are found, returns ``None``.
        """
        for space in getattr(self, spaces):
            if space[0] == template:
                if specs is None or space[1] == specs:
                    return space


def _filter_std(space):
    return space not in NONSTANDARD_REFERENCES


def _filter_nstd(space):
    return not _filter_std(space)


def _filter_surf(space):
    return space.startswith('fs')


def _filter_vol(space):
    return not _filter_surf(space)


def _filter_std_vols(space):
    return space not in NONSTANDARD_REFERENCES and not _filter_surf(space)
