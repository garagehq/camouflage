###${_STUB_IMPORTS} # noqa
from depthai import *
###${_STUB_IMPORTS} # noqa

from math import dist
import marshal
import time

fps = 0.0  ###${_fps}# noqa

def cb(message):
    node.warn(f"{message.getSequenceNum()}")

node.warn(str(node.io))
i = 0
cb_id = None
while True:
    # if not cb_id:
    #     cb_id = node.io['from_producer'].addCallback(cb)

    test = ImgFrame(i)
    test.setSequenceNum(i)

    res =node.io['from_producer'].send(test)
    node.warn(f"{res}")
    time.sleep(4 / fps)

    i += 1