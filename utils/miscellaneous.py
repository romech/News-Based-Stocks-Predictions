import os
import random
import tempfile


def get_tempfile_name():
    return os.path.join(tempfile.gettempdir(), "tempfile-" + format(random.getrandbits(64), 'x'))
