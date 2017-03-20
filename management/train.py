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
    if fake:
        code = SUCCESS
    else:
        code = subprocess.call(call)
    if code == SUCCESS:
        click.secho('Sucess starting gpu tensorflow container', fg='green')
        click.secho('+\n++\n+++ Copying necessary files to container for execution...')
        call = ['sudo', 'docker', 'scp', 'notebook/{}.py', '/']
        if fake:
            code = SUCCESS
        else:
            copty_to_container('notebook/{}.py'.format(model), '/')
            copty_to_container('numerai_training_data.csv', '/')
            copty_to_container('numerai_tournament_data.csv', '/')
    else:
        click.secho('Failure starting gpu tensorflow container', fg='red')
    call = ['jupyter', 'nbconvert', '--to', 'python', 'notebook/'+model+'.ipynb']
    if fake:
        code = SUCCESS
    else:
        code = subprocess.call(call)
    if code == SUCCESS:
        click.secho('Sucess converting ipynb file to py, ready to run', fg='green')
    else:
        click.secho('Failure converting ipynb file to py, not ready to run', fg='red')


def copty_to_container(fi, location):
    call = ['sudo', 'docker', 'scp', fi, 'gpu_tensorflow:{}'.format(location)]
    code = subprocess.call(call)
    if code == SUCCESS:
        click.secho('Successfully copied!')
    return code


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
