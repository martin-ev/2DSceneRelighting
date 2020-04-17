import os
import pwd
import signal
import subprocess
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


TENSORBOARD_LOG_DIR = '/ivrldata1/students/team6/runs'  # specify where logs should be stored
TENSORBOARD_PORT = 6006  # tensorboard will run on localhost on this port


def generate_run_name(label=None):
    """
    Generates name for the run monitored on tensorboard.
    Optional label can contain additional description to identify the run easier.
    @param label: optional name to be added to the run identifier
    @return: formatted run name
    """
    time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    owner = get_logged_in_user_name()
    return f'{time}_{owner}' if label is None else f'{time}_{owner}_{label}'


def get_logged_in_user_name():
    """
    Returns login of the logged in user
    @return: login of the logged in user
    """
    return pwd.getpwuid(os.getuid()).pw_name


def setup_summary_writer(run_name=''):
    """
    Creates a writer object used to update tensorboard content.
    @param run_name: (optional) name for the experiment with which it's run will be labeled
    @return: a SummaryWriter object
    """
    run_name = generate_run_name(label=run_name)
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR, run_name))
    print(f'Created summary writer for run {run_name}')
    return writer


def start_tensorboard_process():
    """
    Starts tensorboard process in the background.
    @return: object containing information about the tensorboard process such as its PID
    """
    command = f'tensorboard --logdir {TENSORBOARD_LOG_DIR} --port {TENSORBOARD_PORT} &'
    # see https://stackoverflow.com/a/19152273, https://stackoverflow.com/a/9935511 and
    # https://stackoverflow.com/a/4791612
    tensorboard_process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    print(f'Tensorboard is now running. To access it on your local computer at localhost:{TENSORBOARD_PORT} run:')
    print(f'    ssh - N - f - L localhost:{TENSORBOARD_PORT}:localhost:{TENSORBOARD_PORT} '
          f'{get_logged_in_user_name()}@iccluster134.iccluster.epfl.ch')
    return tensorboard_process


def stop_tensorboard_process(tensorboard_process):
    """
    Terminates the process group associated with the given process
    @param tensorboard_process: tensorboard process object that was returned by start_tensorboard_process()
    @return: None
    """
    os.killpg(os.getpgid(tensorboard_process.pid), signal.SIGTERM)  # see https://stackoverflow.com/a/4791612
