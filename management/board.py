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
def board(folder):
    """Run tensorboard on folder."""
    click.secho("+\n++\n+++ Running tensorboard for folder {}...".format(folder))
    call = "docker exec gpu_tensorflow bash -c 'tensorboard --logdir {}' &".format(folder)
    code = subprocess.call(call, shell=True)
    if code == SUCCESS:
        click.secho('Successfully began tensorboard', fg='green')
    else:
        click.secho('Failure starting tensorboard', fg='red')


if __name__ == '__main__':
    board()
