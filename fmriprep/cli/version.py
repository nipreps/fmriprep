# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Version CLI helpers."""

from pathlib import Path
from datetime import datetime
import requests
from .. import __version__

RELEASE_EXPIRY_DAYS = 14
DATE_FMT = '%Y%m%d'


def check_latest():
    """Determine whether this is the latest version."""
    from packaging.version import Version, InvalidVersion

    latest = None
    date = None
    outdated = None
    cachefile = Path.home() / '.cache' / 'fmriprep' / 'latest'
    try:
        cachefile.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        cachefile = None

    if cachefile and cachefile.exists():
        try:
            latest, date = cachefile.read_text().split('|')
        except Exception:
            pass
        else:
            try:
                latest = Version(latest)
                date = datetime.strptime(date, DATE_FMT)
            except (InvalidVersion, ValueError):
                latest = None
            else:
                if abs((datetime.now() - date).days) > RELEASE_EXPIRY_DAYS:
                    outdated = True

    if latest is None or outdated is True:
        try:
            response = requests.get(url='https://pypi.org/pypi/fmriprep/json', timeout=1.0)
        except Exception:
            response = None

        if response and response.status_code == 200:
            versions = [Version(rel) for rel in response.json()['releases'].keys()]
            versions = [rel for rel in versions if not rel.is_prerelease]
            if versions:
                latest = sorted(versions)[-1]
        else:
            latest = None

    if cachefile is not None and latest is not None:
        try:
            cachefile.write_text('|'.join(('%s' % latest, datetime.now().strftime(DATE_FMT))))
        except Exception:
            pass

    return latest


def is_flagged():
    """Check whether current version is flagged."""
    # https://raw.githubusercontent.com/poldracklab/fmriprep/master/.versions.json
    flagged = tuple()
    try:
        response = requests.get(url="""\
https://raw.githubusercontent.com/poldracklab/fmriprep/master/.versions.json""", timeout=1.0)
    except Exception:
        response = None

    if response and response.status_code == 200:
        flagged = response.json().get('flagged', {}) or {}

    if __version__ in flagged:
        return True, flagged[__version__]

    return False, None
