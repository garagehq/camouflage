from typing import Callable

from depthai import *

class IOOutput:
    send: Callable
class IOInput:
    get: Callable
    tryGet: Callable

node = node