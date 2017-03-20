"""
Script runs flake8 linter which checks code style for conformance.

Specifically the linter checks for PEP08 code compliance as well as for
PEP257 docstring compliance.

:copyright: (c) 2016 Pinn Technologies, Inc.
:license: All rights reserved
"""

import click
import subprocess
import re
from os import listdir

SUCCESS = 0


@click.command()
@click.option('--folder', default='ckpt/linear')
def clean(folder):
    """Call the flake8 and pydocstyle linter on the entire project."""
    click.secho("+\n++\n+++ Cleaning checkpoint folder {}...".format(folder))
    evals(folder)
    events(folder)
    meta(folder)


def evals(folder):
    """Remove all evals."""
    call = "sudo rm -rf {}/eval".format(folder)
    code = subprocess.call(call, shell=True)
    if code == SUCCESS:
        click.secho('Successfully removed eval folder', fg='green')
    else:
        click.secho('Failure removing eval folder', fg='red')


def events(folder):
    """Remove all events."""
    call = "sudo rm -f {}/events.*".format(folder)
    code = subprocess.call(call, shell=True)
    if code == SUCCESS:
        click.secho('Successfully removed events', fg='green')
    else:
        click.secho('Failure removing events', fg='red')


def meta(folder):
    """Traverse folder and remove all but largest train for checkpoint."""
    ff = listdir(folder)
    ckpts = []
    for fi in ff:
        if 'meta' in fi:
            ckpts.extend(re.findall(r'\d+', fi))
    ckpts.sort(key=int)
    del ckpts[-1]
    for ckpt in ckpts:
        call = "sudo rm -f {}/model.ckpt-{}.*".format(folder, ckpt)
        code = subprocess.call(call, shell=True)
        if code != SUCCESS:
            click.secho('Failure removing model ckpts', fg='red')
    click.secho('Successfully removed all but latest checktpoint', fg='green')

if __name__ == '__main__':
    clean()
