###${_STUB_IMPORTS} # noqa
from dai_available_imports import *
###${_STUB_IMPORTS} # noqa

from math import dist

pad_h = 0  ###${_pad_h} # noqa
img_h = 0  ###${_img_h} # noqa
img_w = 0  ###${_img_w} # noqa
frame_size = 0  ###${_frame_size} # noqa

# For Movenet smart cropping, defines the default crop region (pads the full image from both sides to make it a
# square image) Used when the algorithm cannot reliably determine the crop region from the previous frame.
init_crop_region = {'x_min': 0, 'y_min': -pad_h, 'x_max': frame_size, 'y_max': -pad_h + frame_size, 'size': frame_size}
# noinspection DuplicatedCode
crop_region = init_crop_region

body_score_thresh = 0.0  ###${_body_score_thresh}# noqa
hands_up_only = 0.0  ###${_hands_up_only}# noqa

pd_score_thresh = 0.0  ###${_pd_score_thresh}# noqa
lm_score_thresh = 0.0  ###${_lm_score_thresh}# noqa
single_hand_tolerance_thresh = 0.0  ###${_single_hand_tolerance_thresh}# noqa

# noinspection DuplicatedCode
BODY_KP = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

torso_joints = [BODY_KP["left_hip"], BODY_KP["right_hip"], BODY_KP["left_shoulder"], BODY_KP["right_shoulder"]]


# BufferMgr is used to statically allocate buffers once
# (replace dynamic allocation).
# These buffers are used for sending the result to host
class BufferMgr:
    def __init__(self):
        # noinspection SpellCheckingInspection
        self._bufs = {}

    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
            ###${_TRACE2} (f"New buffer allocated: {size}") # noqa
        return buf


buffer_mgr = BufferMgr()


# noinspection DuplicatedCode
def determine_torso_and_body_range(x, y, scores, center_x, center_y):
    # Calculates the maximum distance from each keypoints to the center location.
    # The function returns the maximum distances from the two sets of keypoints:
    # full 17 keypoints and 4 torso keypoints. The returned information will be
    # used to determine the crop size. See determine_crop_region for more detail.

    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint_idx in torso_joints:
        dist_y = abs(center_y - y[joint_idx])
        dist_x = abs(center_x - x[joint_idx])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint_idx in range(17):
        if scores[joint_idx] < body_score_thresh:
            continue
        dist_y = abs(center_y - y[joint_idx])
        dist_x = abs(center_x - x[joint_idx])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y
        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


# noinspection DuplicatedCode
def determine_crop_region(scores, x, y):
    # Determines the region to crop the image for the model to run inference on.
    # The algorithm uses the detected joints from the previous frame to estimate
    # the square region that encloses the full body of the target person and
    # centers at the midpoint of two hip joints. The crop size is determined by
    # the distances between each joint and the center point.
    # When the model is not confident with the four torso-based joint predictions, the
    # function returns a default crop which is the full image padded to square.

    torso_visible = ((scores[BODY_KP["left_elbow"]] > body_score_thresh or
                      scores[BODY_KP["right_elbow"]] > body_score_thresh) and
                     (scores[BODY_KP["left_shoulder"]] > body_score_thresh or
                      scores[BODY_KP["right_shoulder"]] > body_score_thresh))

    if torso_visible:
        center_x = (x[11] + x[12]) // 2
        center_y = (y[11] + y[12]) // 2
        max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = determine_torso_and_body_range(x, y,
                                                                                                              scores,
                                                                                                              center_x,
                                                                                                              center_y)
        crop_length_half = max(max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2,
                               max_body_xrange * 1.2)
        crop_length_half = int(
            round(min(crop_length_half, max(center_x, img_w - center_x, center_y, img_h - center_y))))
        crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

        if crop_length_half > max(img_w, img_h) / 2:
            return init_crop_region
        else:
            crop_length = crop_length_half * 2
            return {'x_min': crop_corner[0], 'y_min': crop_corner[1], 'x_max': crop_corner[0] + crop_length,
                    'y_max': crop_corner[1] + crop_length, 'size': crop_length}
    else:
        return init_crop_region


def movenet_postprocess(body, crop_region):
    # TODO This will all change
    size = crop_region['size']
    x_min = crop_region['x_min']
    y_min = crop_region['y_min']
    scores = []
    x = []
    y = []
    for i in range(17):
        xn = body[3 * i + 1]
        yn = body[3 * i]
        scores.append(body[3 * i + 2])
        # x any are the body keypoint coordinates in the source image in pixels
        x.append(int(x_min + xn * size))
        y.append(int(y_min + yn * size))

    next_crop_region = determine_crop_region(scores, x, y)
    return x, y, scores, next_crop_region


# noinspection DuplicatedCode
def estimate_focus_zone_size(scale=1.0):
    segments = [
        ("left_shoulder", "left_elbow", 2.3),
        ("left_elbow", "left_wrist", 2.3),
        ("left_shoulder", "left_hip", 1),
        ("left_shoulder", "right_shoulder", 1.5),
        ("right_shoulder", "right_elbow", 2.3),
        ("right_elbow", "right_wrist", 2.3),
        ("right_shoulder", "right_hip", 1),
    ]
    lengths = []
    for s in segments:
        id0 = BODY_KP[s[0]]
        id1 = BODY_KP[s[1]]
        if body_scores[id0] > body_score_thresh and body_scores[id1] > body_score_thresh:
            l = dist((body_x[id0], body_y[id0]), (body_x[id1], body_y[id1]))
            lengths.append(l)
    if lengths:
        if (body_scores[BODY_KP["left_hip"]] < body_score_thresh and
                body_scores[BODY_KP["right_hip"]] < body_score_thresh or
                body_scores[BODY_KP["left_shoulder"]] < body_score_thresh and
                body_scores[BODY_KP["right_shoulder"]] < body_score_thresh):
            coefficient = 1.0
        else:
            coefficient = 1.0
        return 2 * int(coefficient * scale * max(lengths) / 2)  # The size is made even
        # return coefficient * scale * max(lengths)
    else:
        return 0


def zone_from_center_size(x, y, size):
    # Return zone [left, top, right, bottom]
    # from zone center (x,y) and zone size (the zone is square).
    half_size = size // 2
    size = half_size * 2
    if size > img_w:
        x = img_w // 2
    x1 = x - half_size
    if x1 < 0:
        x1 = 0
    elif x1 + size > img_w:
        x1 = img_w - size
    x2 = x1 + size
    if size > img_h:
        y = img_h // 2
    y1 = y - half_size
    if y1 < -pad_h:
        y1 = -pad_h
    elif y1 + size > img_h + pad_h:
        y1 = img_h + pad_h - size
    y2 = y1 + size
    return [x1, y1, x2, y2]


def get_one_hand_zone(hand_label):
    # Return the zone [left, top, right, bottom] around the hand given by its label "hand_label" ("left" or "right")
    # Values are expressed in pixels in the source image C.S.
    # If the wrist keypoint is not visible, return None.
    # If self.hands_up_only is True, return None if wrist keypoint is below elbow keypoint.
    wrist_kp = hand_label + "_wrist"
    id_wrist = BODY_KP[wrist_kp]
    if body_scores[id_wrist] < body_score_thresh:
        return None
    x = body_x[id_wrist]
    y = body_y[id_wrist]
    if hands_up_only:
        # TODO reimplement this based on the xyz readings rather than 2D pixel readings, the overhead perspective isn't
        #  compatible with the assumptions of this algorithm
        # We want to detect only hands where the wrist is above the hip (when visible)
        elbow_kp = hand_label + "_hip"
        id_elbow = BODY_KP[elbow_kp]
        if body_scores[id_elbow] > body_score_thresh and \
                body_y[id_elbow] < y:
            return None
    # Let's evaluate the size of the focus zone
    size = estimate_focus_zone_size()
    if size == 0: return [0, -pad_h, frame_size, frame_size - pad_h]
    return zone_from_center_size(x, y, size)


def get_focus_zone():
    if hand_label == "group":
        zone_l = get_one_hand_zone("left")
        if zone_l:
            zoner = get_one_hand_zone("right")
            if zoner:
                xl1, yl1, xl2, yl2 = zone_l
                xr1, yr1, xr2, yr2 = zoner
                x1 = min(xl1, xr1)
                y1 = min(yl1, yr1)
                x2 = max(xl2, xr2)
                y2 = max(yl2, yr2)
                # Center (x,y)
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                size_x = x2 - x1
                size_y = y2 - y1
                size = max(size_x, size_y)
                zone_hand = [zone_from_center_size(x, y, size), "group"]
            else:
                zone_hand = [zone_l, "left"]
        else:
            zoner = get_one_hand_zone("right")
            if zoner:
                zone_hand = [zoner, "right"]
            else:
                zone_hand = [None, None]
    return zone_hand


while True:
    # noinspection DuplicatedCode
    cfg_pre_body = ImageManipConfig()
    points = [
        [crop_region['x_min'], crop_region['y_min']],
        [crop_region['x_max'] - 1, crop_region['y_min']],
        [crop_region['x_max'] - 1, crop_region['y_max'] - 1],
        [crop_region['x_min'], crop_region['y_max'] - 1]]
    point2fList = []
    for p in points:
        pt = Point2f()
        pt.x, pt.y = p[0], p[1]
        point2fList.append(pt)
    cfg_pre_body.setWarpTransformFourPoints(point2fList, False)
    body_input_length = None  ###${_body_input_length} # noqa
    cfg_pre_body.setResize(body_input_length, body_input_length)
    cfg_pre_body.setFrameType(ImgFrame.Type.RGB888p)
    node.io['pre_body_manip_cfg'].send(cfg_pre_body)
    ###${_TRACE2} ("Manager sent thumbnail config to pre_body manip") # noqa
    # Wait for the body detection result
    body = node.io['from_body_nn'].get().getLayerFp16("Identity")

    # Extract body keypoints and calculate smart crop for the next frame
    body_x, body_y, body_scores, crop_region = movenet_postprocess(body, crop_region)

    # Calculate pre focus zone(s)
    zone, hand_label = get_focus_zone()
    if zone:
        x_min, y_min, x_max, y_max = zone
        ###${_TRACE1} (f"Body pre focusing zone: ({x_min}, {y_min}), ({x_max}, {y_max})") # noqa
        # noinspection DuplicatedCode
        points = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        point2fList = []
        for p in points:
            pt = Point2f()
            pt.x, pt.y = p[0], p[1]
            point2fList.append(pt)
        cfg_pre_pd = ImageManipConfig()
        cfg_pre_pd.setWarpTransformFourPoints(point2fList, False)
        cfg_pre_pd.setResize(128, 128)
        send_new_frame_to_branch = 1
    else:
        ###${_TRACE1} (f"Body pre focusing zone: None") # noqa
        x_serial = marshal.dumps([1, 0])
        b = buffer_mgr(len(x_serial))
        b.setData(x_serial)
        node.io['results_no_hands'].send(b)
        nb_hands_in_previous_frame = 0
        continue
