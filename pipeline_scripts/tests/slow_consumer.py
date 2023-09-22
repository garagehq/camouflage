###${_STUB_IMPORTS} # noqa
from depthai import *
###${_STUB_IMPORTS} # noqa

from math import dist
import marshal
import time

fps = 0.0  ###${_fps}# noqa

while True:
    frame = node.io['from_producer'].get()
    # x = marshal.loads(message.getData())
    node.warn(f"received message: {frame.getSequenceNum()}")
    time.sleep(10 / fps)