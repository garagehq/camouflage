###${_STUB_IMPORTS} # noqa
from ..import_stub import * # noqa
from ..import_stub import node
from typing import Dict, Literal, Union
proto: Dict[Literal[
    "detection_results",
    "TMP",
], Union[IOOutput, IOInput]] = {}
node.io = proto
###${_STUB_IMPORTS} # noqa

script_name = ""  ###${_NAME} # noqa
fps = 0.0  ###${_fps}# noqa

bbox_confidence_thresh = 0.0  ###${_bbox_confidence_thresh}# noqa
lm_score_thresh = 0.0  ###${_lm_score_thresh}# noqa


while True:
    bboxes: NNData = node.io['detection_results'].get()
