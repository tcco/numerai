"""
Script that imports all relavant classes to a new python shell.

:copyright: (c) 2016 Pinn Technologies, Inc.
:license: All rights reserved
"""


import click
import subprocess


@click.command()
def shell():
    """Start a convenience python shell for the application."""
    subprocess.call(['python', '-i', './management/shell_env.py'])

if __name__ == '__main__':
    shell()
