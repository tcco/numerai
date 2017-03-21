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
@click.option('--model', default='linear')
@click.option('--nohup', default=False, is_flag=True)
@click.option('--steps', default='5000')
def train(fake, model, nohup, steps):
    """Subprocess python shell."""
    if nohup:
        choice = 'd'
    else:
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
        code = bootup(model, fake)
    else:
        code = SUCCESS

    if choice in ['b', 'd'] and code == SUCCESS:
        if not nohup:
            steps = click.prompt('Number of steps for training?', type=str)
        code = _train(model, steps, fake)

    if choice in ['c', 'd'] and code == SUCCESS:
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
    else:
        click.secho(
            '+\n++\n+++ That did not work.... Lets try the old-school way...')
        call = ['sudo', 'nvidia-docker', 'run', '-tid', '--name=gpu_tensorflow',
                '-e', 'PASSWORD=Pinncode206', '-p', '8754:8888', '-p', '6006:6006',
                '-v', '/home/ubuntu/numerai:/notebooks', 'tensorflow/tensorflow:latest-gpu']
        code = subprocess.call(call)
        if code == SUCCESS:
            click.secho(
                'Success starting gpu tensorflow container', fg='green')
        else:
            click.secho('Failure starting gpu tensorflow container', fg='red')
            return code

    click.secho('+\n++\n+++ Installing necessary packages...')
    call = ['docker', 'exec', 'gpu_tensorflow', 'bash', '-c',
            'pip install -r requirements.txt --quiet --user']
    code = subprocess.call(call)
    if code == SUCCESS:
        click.secho('Success installing necessary packages', fg='green')
    else:
        click.secho('Failure installing necessary packages', fg='red')
        return code

    click.secho('+\n++\n+++ Converting notebook .ipynb files to py...')
    call = ['jupyter', 'nbconvert', '--to',
            'python', 'notebook/' + model + '.ipynb']
    if fake:
        code = SUCCESS
    else:
        code = subprocess.call(call)
    if code == SUCCESS:
        click.secho(
            'Sucess converting ipynb file to py, ready to run', fg='green')
    else:
        click.secho(
            'Failure converting ipynb file to py, not ready to run', fg='red')
    return code


def _train(model, steps, fake):
    """Train given model."""
    click.secho('+\n++\n+++ Lets begin training...')
    click.secho('+\n++\n+++ Using model {}...'.format(model))
    click.secho('+\n++\n+++ Training for {} steps...'.format(steps))
    call = ['docker', 'exec', 'gpu_tensorflow', 'bash', '-c',
            'python notebook/{}.py --steps {}'.format(model, steps)]
    if fake:
        code = SUCCESS
    else:
        code = subprocess.call(call)
    if code == SUCCESS:
        click.secho(
            'Sucess training! Checkout the logs /results at data/loggy.logs', fg='green')
    else:
        click.secho('Failure training :(', fg='red')
    return code


def shutdown(fake):
    """Shutdown computer."""
    click.secho('Cleaning checkpoint folder...', fg='red')
    call = ['./manage', 'clean']
    subprocess.call(call)
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
