"""
Deployment manage commands.

:copyright: (c) 2016 Pinn
:license: All rights reserved

"""

import click
import subprocess

SUCCESS = 0


@click.command()
@click.option('--fake', default=False, is_flag=True)
@click.option('--model', default='dnnlinear')
def train(fake, model):
    """Subprocess python shell."""
    click.secho('\nChoose training proces:')
    click.secho('a) Boot Up')
    click.secho('b) Train')
    click.secho('c) Shutdown')
    click.secho('d) All of the Above')

    choice = click.prompt(click.style('default is b'),
                          type=click.Choice(['a', 'b', 'c', 'd']),
                          default='b',
                          show_default=False)

    if choice in ['a', 'd']:
        bootup(model, fake)

    if choice in ['b', 'd']:
        steps = click.prompt('How many steps of training should we commit?', type=str)
        _train(model, steps, fake)

    if choice in ['c', 'd']:
        shutdown(fake)


def bootup(model, fake):
    """Necessary steps for booting up AWS instance."""
    click.secho('+\n++\n+++ Lets boot up...')
    click.secho('+\n++\n+++ Using model {}...'.format(model))
    call = ['sudo', 'docker', 'start', 'gpu_tensorflow']
    call_2 = ['jupyter', 'nbconvert', '--to', 'python', 'notebook/'+model+'.ipynb']
    if fake:
        code = SUCCESS
        code_2 = SUCCESS
    else:
        code = subprocess.call(call)
        code_2 = subprocess.call(call_2)
    if code == SUCCESS and code_2 == SUCCESS:
        click.secho('Sucess booting up', fg='green')
    else:
        click.secho('Failure booting up', fg='red')


def _train(model, steps, fake):
    """Train given model."""
    click.secho('+\n++\n+++ Lets begin training...')
    click.secho('+\n++\n+++ Using model {}...'.format(model))
    click.secho('+\n++\n+++ Training for {} steps...'.format(steps))
    call = ['python', 'notebook/'+model+'.py', '--steps', steps]
    if fake:
        code = SUCCESS
    else:
        code = subprocess.call(call)
    if code == SUCCESS:
        click.secho('Sucess training! Checkout the logs /results at data/loggy.logs', fg='green')
    else:
        click.secho('Failure training :(', fg='red')


def shutdown(fake):
    """Shutdown computer."""
    click.secho('Shutting down... Now...', fg='red')
    click.secho('Bye... Bye... Bye...', fg='red')
    call = ['sudo', 'shutdown', 'now']
    if fake:
        code = SUCCESS
    else:
        code = subprocess.call(call)
    if code == SUCCESS:
        click.secho('Sucess shutting down', fg='green')
    else:
        click.secho('Failure shutting down', fg='red')


if __name__ == '__main__':
    train()
