"""
Script runs flake8 linter which checks code style for conformance.

Specifically the linter checks for PEP08 code compliance as well as for
PEP257 docstring compliance.

:copyright: (c) 2016 Pinn Technologies, Inc.
:license: All rights reserved
"""

import click
import subprocess


@click.command()
def lint():
    """Call the flake8 and pydocstyle linter on the entire project."""
    # flake8
    flake8_return_value = subprocess.call(['flake8',
                                           '.',
                                           '--exclude=./.venv,./docs'])
    if flake8_return_value != 0:
        return flake8_return_value
    # pydocstyle
    return subprocess.call(['pydocstyle', '.'])

if __name__ == '__main__':
    lint()
