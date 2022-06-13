import os
import sys


def append_sys_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build = os.path.join(root, "build")
    if build not in sys.path:
        sys.path.append(build)


append_sys_path()
