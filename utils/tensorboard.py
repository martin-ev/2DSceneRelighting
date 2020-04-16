import pwd
import os
from datetime import datetime


def generate_run_name(label=None):
    """
    Generates name for the run monitored on tensorboard.
    Optional label can contain additional description to identify the run easier.
    """
    time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    owner = pwd.getpwuid(os.getuid()).pw_name  # get login of the person who ran the script
    if label is None:
        return f'{time}_{owner}'
    else:
        return f'{time}_{owner}_{label}'
