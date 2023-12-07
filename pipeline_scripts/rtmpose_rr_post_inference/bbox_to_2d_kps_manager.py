###${_STUB_IMPORTS} # noqa
from ..import_stub import *  # noqa
from ..import_stub import node
from typing import Dict, Literal, Union

proto: Dict[Literal[
    "detection_results",
    "pre_pose_manip_cfg",
    "TMP",
    "post_center_size_angles",
], Union[IOOutput, IOInput]] = {}
node.io = proto
###${_STUB_IMPORTS} # noqa
from math import sin, cos, atan2, degrees

script_name = ""  ###${_NAME} # noqa
fps = 0.0  ###${_fps}# noqa

bbox_confidence_thresh = 0.0  ###${_bbox_confidence_thresh}# noqa
lm_score_thresh = 0.0  ###${_lm_score_thresh}# noqa
max_bodies = 0.0  ###${_max_bodies}# noqa

while True:
    detections: ImgDetections = node.io['detection_results'].get()
    node.warn(f"det len{len(detections.detections)}")
    for i, detection in enumerate(detections.detections):
        detection: ImgDetection = detection
        if detection.confidence < bbox_confidence_thresh:
            continue
        cfg = ImageManipConfig()
        rr = RotatedRect()
        rr.size.width, rr.size.height = detection.xmax - detection.xmin, detection.ymax - detection.ymin
        rr.center.x, rr.center.y = detection.xmin + (rr.size.width / 2), detection.ymin + (rr.size.height / 2)
        norm_rot = atan2(rr.center.y - frame_size_y // 2, rr.center.x - frame_size_x // 2)
        rr.angle = degrees(norm_rot)
        cfg.setCropRotatedRect(rr, False)
        node.io['pre_pose_manip_cfg'].send(cfg)
        modified_center = [e * 10 for e in [rr.center.x, rr.center.y]]
        modified_size = [e * 10 for e in [rr.size.width, rr.size.height]]
        # convert range -1 to 1 into 0 to 20000, which can then be stored as uint8's and sent to the model, which
        #   reverses this process to get an FP16 lossless-ly(+/-.0001 due to float rounding errors?)
        modified_angles = [(e + 1) * 10000 for e in [cos(rr.angle), sin(rr.angle)]]
        csa_uint16 = modified_center + modified_size + modified_angles
        csa_uint8 = [list(int(val).to_bytes(2, byteorder='little')) for val in csa_uint16]
        csa_uint8_flat = [item for sublist in csa_uint8 for item in sublist]
        node.warn(f"transferring buffer with data: {csa_uint8}\t built from {csa_uint16}")
        csa_buff = Buffer()
        csa_buff.setData(csa_uint8_flat)
        node.io['post_center_size_angles'].send(csa_buff)
