# noinspection DuplicatedCode
# Modified from HandTrackerBpfEdge
import json
import marshal
import sys
from pathlib import Path
from string import Template
from typing import Literal, get_args, Tuple, cast, Union, Dict

import cv2
import depthai.node as n
import numpy as np
from depthai import Pipeline, CameraBoardSocket, Device, CameraSensorType, MonoCameraProperties, \
    ColorCameraProperties, OpenVINO, ImgFrame
import mediapipe_utils as mpu
from FPS import FPS

# noinspection DuplicatedCode
SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(
    SCRIPT_DIR / "custom_models/PDPostProcessing_top2_sh1.blob"
)
MOVENET_LIGHTNING_MODEL = str(
    SCRIPT_DIR / "models/movenet_singlepose_lightning_U8_transpose.blob"
)
MOVENET_THUNDER_MODEL = str(
    SCRIPT_DIR / "models/movenet_singlepose_thunder_U8_transpose.blob"
)
TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR / "template_manager_script_bpf_solo.py")
TEMPLATE_MANAGER_SCRIPT_DUO = str(SCRIPT_DIR / "template_manager_script_bpf_duo.py")
TEMPLATE_MANAGER_SCRIPT_MULTI_BODY = str(
    SCRIPT_DIR / "template_manager_script_bpf_multi_body.py"
)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


TOF = Literal['oak_d_sr_poe']
SRBase = Literal['oak_d_sr']
RGBStereoPair = Literal[SRBase, TOF]
MonoStereoPair = Literal['oak_d_s2', 'oak_d_lite', 'oak_d_pro']
DisparityDepth = Literal[SRBase, MonoStereoPair]

DeviceModel = Literal[TOF, RGBStereoPair, MonoStereoPair]

MovenetModel = Literal["lightning", "thunder"]

RGBFullResolution = Literal["full"]
RGBUltraResolution = Literal["ultra"]
ResolutionType = Literal[RGBFullResolution, RGBUltraResolution]

SRRGBResolutionDim = Tuple[Literal[1280], Literal[800]]
RGBFullResolutionDim = Tuple[Literal[1920], Literal[1080]]
RGBUltraResolutionDim = Tuple[Literal[3840], Literal[2160]]
SupportedResolution = Union[SRRGBResolutionDim, RGBFullResolutionDim, RGBUltraResolutionDim]

TraceLevel = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class DeviceCapability:
    def __init__(self, res: SupportedResolution, mono_stereo: bool):
        self.res, self.mono_stereo = res, mono_stereo
        self.res_type = MonoCameraProperties.SensorResolution.THE_400_P \
            if mono_stereo else ColorCameraProperties.SensorResolution.THE_800_P

    def run_constructor(self, pipeline: Pipeline) -> n.ColorCamera | n.MonoCamera:
        con = pipeline.createMonoCamera if self.mono_stereo else pipeline.createColorCamera
        return con()

    def correct_preview_size(self, cam: n.ColorCamera | n.MonoCamera):
        if self.mono_stereo:
            return
        cam.setPreviewSize(320, 320)




# noinspection PyTypeChecker
device_cfgs: Dict[DeviceModel, DeviceCapability] = {
    "oak_d_sr_poe": DeviceCapability(
        tuple(get_args(item)[0] for item in get_args(SRRGBResolutionDim)),
        False
    ),
    "oak_d_sr": DeviceCapability(
        tuple(get_args(item)[0] for item in get_args(SRRGBResolutionDim)),
        False
    ),
    "oak_d_s2": DeviceCapability(
        tuple(get_args(item)[0] for item in get_args(RGBFullResolutionDim)),
        True
    ),
    "oak_d_lite": DeviceCapability(
        tuple(get_args(item)[0] for item in get_args(RGBFullResolutionDim)),
        True
    ),
    "oak_d_pro": DeviceCapability(
        tuple(get_args(item)[0] for item in get_args(RGBFullResolutionDim)),
        True
    ),
}


# noinspection DuplicatedCode
class TGTTracker:
    """
    Mediapipe Hand Tracker for depthai
    Arguments:
    - `input_src`: frame source,
                    - `rgb` or None: OAK* internal color camera,
                    - `rgb_laconic`: same as `rgb` but without sending the frames to the host (Edge mode only),
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
                    In edge mode, only `rgb` and `rgb_laconic` are possible
    - `pd_model`: palm detection model blob file,
    - `pd_score`: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - `pd_nms_thresh`: NMS threshold.
    - `use_lm`: when True, run landmark model, otherwise, only the palm detection model is run
    - `lm_model`: landmark model. Either:
                    - 'full' for LANDMARK_MODEL_FULL,
                    - 'lite' for LANDMARK_MODEL_LITE,
                    - 'sparse' for LANDMARK_MODEL_SPARSE,
                    - a path of a blob file.
    - `lm_score_thresh`: confidence score to determine whether landmark prediction is reliable
                    (a float between 0 and 1).
    - `use_world_landmarks`: boolean. The landmark model yields 2 types of 3D coordinates:
                    - coordinates expressed in pixels in the image, always stored in hand.landmarks,
                    - coordinates expressed in meters in the world, stored in hand.world_landmarks
                    only if `use_world_landmarks` is True.
    - `pp_model`: path to the detection post-processing model,
    - `solo`: boolean, when True detect one hand max (much faster since we run the pose detection model only if no hand
                    was detected in the previous frame) On edge mode, always True
    - `xyz` : boolean, when True calculate the (x, y, z) coords of the detected palms.
    - `crop` : boolean which indicates if square cropping on source images is applied or not
    - `internal_fps` : when using the internal color camera as input source, set its FPS to this value
                    (calling setFps()).
    - `resolution` : sensor resolution `full` (1920x1080) or `ultra` (3840x2160),
    - `internal_frame_height`: when using the internal color camera, set the frame height (calling setIspScale()).
                    The width is calculated accordingly to height and depends on the value of 'crop'
    - `use_gesture` : boolean, when True, recognize hand poses from a predefined set of poses
                    (ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - `body_model`: Movenet single pose model: `lightning` or `thunder`
    - `body_score_thresh`: Movenet score thresh
    - `hands_up_only`: when using body_pre_focusing, if hands_up_only is True, consider only hands for which
                    the wrist keypoint is above the elbow keypoint.
    - `single_hand_tolerance_thresh` (Duo mode only): In Duo mode, if there is only one hand in a frame,
                    in order to know when a second hand will appear, you need to run the palm detection
                    in the following frames. Because palm detection is slow, you may want to delay
                    the next time you will run it. 'single_hand_tolerance_thresh' is the number of
                    frames during only one hand is detected before palm detection is run again.
    - `lm_nb_threads` : 1 or 2 (default=2), number of inference threads for the landmark model
    - `use_same_image` (Edge Duo mode only) : boolean, when True, use the same image when inferring the landmarks of the
                    2 hands (setReusePreviousImage(True) in the ImageManip node before the landmark model).
                    When True, the FPS is significantly higher but the skeleton may appear shifted on one of the 2
                    hands.
    - `stats` : boolean, when True, display some statistics when exiting.
    - `trace` : int, 0 = no trace, otherwise print some debug messages or show output of ImageManip nodes
            if trace & 1, print application level info like number of palm detections,
            if trace & 2, print lower level info like when a message is sent or received by the manager script node,
            if trace & 4, show in cv2 windows outputs of ImageManip node,
            if trace & 8, save in file tmp_code.py the python code of the manager script node
            Ex: if trace==3, both application and low level info are displayed.
    """

    def __init__(
            self,
            input_src: DeviceModel = 'oak_d_sr_poe',
            laconic=True,
            pd_model=PALM_DETECTION_MODEL,
            pd_score_thresh=0.5,
            use_lm=True,
            lm_model="lite",
            lm_score_thresh=0.3,
            use_world_landmarks=True,
            pp_model=DETECTION_POSTPROCESSING_MODEL,
            max_bodies=4,
            crop=False,
            internal_fps=None,
            resolution: ResolutionType = 'full',
            internal_frame_height: int = 640,
            use_gesture=True,
            # TODO swap to Centernet trained from https://github.com/xingyizhou/CenterNet.git
            body_model: MovenetModel = "thunder",
            body_score_thresh=0.3,
            hands_up_only=False,
            single_hand_tolerance_thresh: int = 10,
            use_same_image=True,
            lm_nb_threads: int = 2,
            stats=False,
            trace: int = 0,
    ):
        self.pd_input_length = 128
        self.lm_input_length = 224
        self.use_lm = use_lm
        if not use_lm:
            print("use_lm=False is not supported in Edge mode.")
            sys.exit()
        self.pd_model = pd_model
        print(f"Palm detection blob     : {self.pd_model}")
        if lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL
        elif lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE
        elif lm_model == "sparse":
            self.lm_model = LANDMARK_MODEL_SPARSE
        else:
            self.lm_model = lm_model
        print(f"Landmark blob           : {self.lm_model}")
        self.pp_model = pp_model
        print(f"PD post processing blob : {self.pp_model}")
        self.max_bodies = max_bodies

        self.body_score_thresh = body_score_thresh
        self.body_input_length = 256
        self.hands_up_only = hands_up_only
        if body_model == "lightning":
            self.body_model = MOVENET_LIGHTNING_MODEL
            self.body_input_length = 192
        else:
            self.body_model = MOVENET_THUNDER_MODEL
        print(f"Body pose blob          : {self.body_model}")

        assert lm_nb_threads in [1, 2]
        self.lm_nb_threads = lm_nb_threads

        print("body_pre_focusing is forced to 'group'")
        self.body_pre_focusing = "group"

        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh

        self.xyz = False
        self.crop = crop
        self.use_world_landmarks = use_world_landmarks

        self.stats = stats
        self.trace = trace
        self.use_gesture = use_gesture
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh
        self.use_same_image = use_same_image

        self.device = Device()

        if input_src in get_args(DeviceModel):
            self.input_device = input_src
            # Camera frames are not sent to the host
            self.laconic = laconic

            if input_src in get_args(RGBStereoPair):
                if resolution != 'full':
                    raise ValueError("Varying resolution for tof sensor has not been implemented, use full")
                self.resolution = tuple(get_args(item)[0] for item in get_args(SRRGBResolutionDim))
            else:
                if resolution == "full":
                    self.resolution = get_args(RGBFullResolutionDim)
                elif resolution == "ultra":
                    self.resolution = get_args(RGBUltraResolutionDim)
                else:
                    print(f"Error: {resolution} is not a valid resolution !")
                    sys.exit()
            print("Sensor resolution:", self.resolution)

            # Check if the device supports stereo
            camera_features = self.device.getConnectedCameraFeatures()

            tof = any([cam for cam in camera_features if CameraSensorType.TOF in cam.supportedTypes])
            l_stereo = next(filter(lambda x: x.socket == CameraBoardSocket.CAM_B, camera_features), False)
            r_stereo = next(filter(lambda x: x.socket == CameraBoardSocket.CAM_C, camera_features), False)
            has_stereo = l_stereo and r_stereo and l_stereo.sensorName == r_stereo.sensorName

            if has_stereo or tof:
                self.xyz = True

            if internal_fps is None:
                base_internal_fps = 20
                lm_frame_cost = {"full": 13, "lite": 3, "sparse": 10}
                xyz_no_tof_cost = 5
                # fmt: off
                self.internal_fps = (
                        base_internal_fps
                        - (lm_frame_cost[lm_model] if lm_model in lm_frame_cost else 0)

                        - (xyz_no_tof_cost if self.xyz else 0)
                )
                # fmt: on
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")

            self.video_fps = (
                self.internal_fps
            )  # Used when saving the output in a video file. Should be close to the real fps

            if input_src in get_args(TOF):
                self.internal_fps = 15

            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(
                    internal_frame_height, self.resolution
                )
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (
                                      int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                                      - self.img_w
                              ) // 2
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(
                    internal_frame_height * self.resolution[0] / self.resolution[1],
                    self.resolution,
                    is_height=False,
                )
                self.img_h = int(
                    round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1])
                )
                self.img_w = int(
                    round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])
                )
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0

            print(
                f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}"
            )
        else:
            print("Invalid input source:", input_src)
            sys.exit()

        # Defines the default crop region (pads the full image from both sides to make it a square image)
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        self.crop_region = mpu.CropRegion(
            -self.pad_w,
            -self.pad_h,
            -self.pad_w + self.frame_size,
            -self.pad_h + self.frame_size,
            self.frame_size,
        )

        # Define and start pipeline
        pipeline = self.create_pipeline()
        self.device.startPipeline(pipeline)
        print(f"Pipeline started - USB speed: {str(self.device.getUsbSpeed()).split('.')[-1]}")

        # Define data queues
        if not self.laconic:
            self.q_video = self.device.getOutputQueue(
                name="cam_out", maxSize=1, blocking=False
            )
        self.q_manager_out = self.device.getOutputQueue(
            name="manager_out", maxSize=1, blocking=False
        )
        # For showing outputs of ImageManip nodes (debugging)
        if self.trace & 4:
            self.q_pre_body_manip_out = self.device.getOutputQueue(
                name="pre_body_manip_out", maxSize=1, blocking=False
            )
            self.q_pre_pd_manip_out = self.device.getOutputQueue(
                name="pre_pd_manip_out", maxSize=1, blocking=False
            )
            self.q_pre_lm_manip_out = self.device.getOutputQueue(
                name="pre_lm_manip_out", maxSize=1, blocking=False
            )

        self.fps = FPS()

        self.nb_frames_body_inference = 0
        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline: Pipeline = Pipeline()
        pipeline.setOpenVINOVersion(version=OpenVINO.Version.VERSION_2021_4)

        # ColorCamera
        print("Creating Color Camera...")
        cam = None
        if self.input_device in get_args(MonoStereoPair):
            cam = pipeline.createColorCamera()
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            if self.resolution == get_args(RGBFullResolutionDim):
                cam.setResolution(ColorCameraProperties.SensorResolution.THE_1080_P)
            else:
                cam.setResolution(ColorCameraProperties.SensorResolution.THE_4_K)
            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)
            else:
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)

        # Define manager script node
        manager_script = pipeline.create(n.Script)
        manager_script.setScript(self.build_manager_script())

        if self.xyz:
            spatial_location_calculator = pipeline.createSpatialLocationCalculator()
            spatial_location_calculator.setWaitForConfigInput(True)
            spatial_location_calculator.inputDepth.setBlocking(False)
            spatial_location_calculator.inputDepth.setQueueSize(1)

            if self.input_device in get_args(DisparityDepth):
                print("Creating XYZ Nodes")
                # For now, RGB needs fixed focus to properly align with depth.
                # The value used during calibration should be used here

                left = device_cfgs[self.input_device].run_constructor(pipeline)
                left.setBoardSocket(CameraBoardSocket.CAM_B)
                left.setResolution(device_cfgs[self.input_device].res_type)
                left.setFps(self.internal_fps)
                left.setInterleaved(False)

                right = device_cfgs[self.input_device].run_constructor(pipeline)
                right.setBoardSocket(CameraBoardSocket.CAM_C)
                right.setResolution(device_cfgs[self.input_device].res_type)
                right.setFps(self.internal_fps)
                right.setInterleaved(False)

                device_cfgs[self.input_device].correct_preview_size(left)
                device_cfgs[self.input_device].correct_preview_size(right)

                stereo = pipeline.createStereoDepth()
                # stereo.inputConfig.possibleDatatypes.append()
                stereo.setConfidenceThreshold(230)
                # LR-check is required for depth alignment
                stereo.setLeftRightCheck(True)
                stereo.setSubpixel(False)  # subpixel True brings latency
                # MEDIAN_OFF necessary in depthai 2.7.2.
                # Otherwise : [critical] Fatal error.
                # Please report to developers.
                # Log: 'StereoSipp' '533'
                # stereo.setMedianFilter(StereoDepthProperties.MedianFilter.MEDIAN_OFF)

                if self.input_device in get_args(MonoStereoPair):
                    calib_data = self.device.readCalibration()
                    calib_lens_pos = calib_data.getLensPosition(CameraBoardSocket.CAM_A)
                    print(f"RGB calibration lens position: {calib_lens_pos}")
                    cam.initialControl.setManualFocus(calib_lens_pos)
                    stereo.setDepthAlign(CameraBoardSocket.CAM_A)
                else:
                    cam = left

                left.preview.link(stereo.left)
                right.preview.link(stereo.right)

                stereo.depth.link(spatial_location_calculator.inputDepth)

            elif self.input_device in get_args(TOF):
                cam_a: n.Camera = pipeline.create(n.Camera)
                # We assume the ToF camera sensor is on port CAM_A
                cam_a.setBoardSocket(CameraBoardSocket.CAM_A)

                tof: n.ToF = pipeline.create(n.ToF)
                cam_a.raw.link(tof.input)

                tof.depth.link(spatial_location_calculator.inputDepth)
            else:
                raise ValueError(f"{self.input_device} is not a supported type")

            manager_script.outputs["spatial_location_config"].link(
                spatial_location_calculator.inputConfig
            )
            spatial_location_calculator.out.link(manager_script.inputs["spatial_data"])

            # Define body pose detection pre-processing: resize preview to (self.body_input_length,
            # self.body_input_length)
            # and transform BGR to RGB

        if not self.laconic:
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
            cam.video.link(cam_out.input)

        print("Creating Body Pose Detection pre processing image manip...")
        pre_body_manip = pipeline.create(n.ImageManip)
        pre_body_manip.setMaxOutputFrameSize(
            self.body_input_length * self.body_input_length * 3
        )
        pre_body_manip.setWaitForConfigInput(True)
        pre_body_manip.inputImage.setQueueSize(1)
        pre_body_manip.inputImage.setBlocking(False)

        cam.preview.link(pre_body_manip.inputImage)

        manager_script.outputs["pre_body_manip_cfg"].link(pre_body_manip.inputConfig)
        # For debugging
        if self.trace & 4:
            pre_body_manip_out = pipeline.createXLinkOut()
            pre_body_manip_out.setStreamName("pre_body_manip_out")
            pre_body_manip.out.link(pre_body_manip_out.input)

        # Define landmark model
        print("Creating Body Pose Detection Neural Network...")
        body_nn = pipeline.create(n.NeuralNetwork)
        body_nn.setBlobPath(Path(self.body_model))
        # body_nn.setNumInferenceThreads(2)
        pre_body_manip.out.link(body_nn.input)
        body_nn.out.link(manager_script.inputs["from_body_nn"])

        # Define palm detection pre-processing: resize preview to (self.pd_input_length, self.pd_input_length)
        print("Creating Palm Detection pre processing image manip...")
        pre_pd_manip = pipeline.create(n.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(
            self.pd_input_length * self.pd_input_length * 3
        )
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs["pre_pd_manip_cfg"].link(pre_pd_manip.inputConfig)

        # For debugging
        if self.trace & 4:
            pre_pd_manip_out = pipeline.createXLinkOut()
            pre_pd_manip_out.setStreamName("pre_pd_manip_out")
            pre_pd_manip.out.link(pre_pd_manip_out.input)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.create(n.NeuralNetwork)
        pd_nn.setBlobPath(Path(self.pd_model))
        pre_pd_manip.out.link(pd_nn.input)

        # Define pose detection post processing "model"
        print("Creating Palm Detection post processing Neural Network...")
        post_pd_nn = pipeline.create(n.NeuralNetwork)
        post_pd_nn.setBlobPath(Path(self.pp_model))
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs["from_post_pd_nn"])

        # Define link to send result to host
        manager_out = pipeline.create(n.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs["host"].link(manager_out.input)

        # Define landmark pre-processing image manip
        print("Creating Hand Landmark pre processing image manip...")
        pre_lm_manip = pipeline.create(n.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(
            self.lm_input_length * self.lm_input_length * 3
        )
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)

        # For debugging
        if self.trace & 4:
            pre_lm_manip_out = pipeline.createXLinkOut()
            pre_lm_manip_out.setStreamName("pre_lm_manip_out")
            pre_lm_manip.out.link(pre_lm_manip_out.input)

        manager_script.outputs["pre_lm_manip_cfg"].link(pre_lm_manip.inputConfig)

        # Define landmark model
        print(
            f"Creating Hand Landmark Neural Network ({self.lm_nb_threads} threads)..."
        )
        lm_nn = pipeline.create(n.NeuralNetwork)
        lm_nn.setBlobPath(Path(self.lm_model))
        lm_nn.setNumInferenceThreads(self.lm_nb_threads)
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs["from_lm_nn"])

        print("Pipeline created.")
        return pipeline

    def build_manager_script(self):
        """
        The code of the scripting node 'manager_script' depends on:
            - the score threshold,
            - the video frame shape,
        So we build this code from the content of the file template_manager_script_*.py,
        which is a python template
        """
        # Read the template
        with open(
                TEMPLATE_MANAGER_SCRIPT_MULTI_BODY,
                "r",
        ) as file:
            template = Template(file.read())

        # Perform the substitution
        code = template.substitute(
            _TRACE1="node.warn" if self.trace & 1 else "#",
            _TRACE2="node.warn" if self.trace & 2 else "#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=self.lm_score_thresh,
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w,
            _frame_size=self.frame_size,
            _crop_w=self.crop_w,
            _IF_XYZ="" if self.xyz else '"""',
            _body_pre_focusing=self.body_pre_focusing,
            _body_score_thresh=self.body_score_thresh,
            _body_input_length=self.body_input_length,
            _hands_up_only=self.hands_up_only,
            _single_hand_tolerance_thresh=self.single_hand_tolerance_thresh,
            _IF_USE_SAME_IMAGE="" if self.use_same_image else '"""',
            _IF_USE_WORLD_LANDMARKS="" if self.use_world_landmarks else '"""',
        )
        # Remove comments and empty lines
        import re

        code = re.sub(r'"{3}.*?"{3}', "", code, flags=re.DOTALL)
        code = re.sub(r"#.*", "", code)
        code = re.sub('\n\s*\n', "\n", code)
        # For debugging
        if self.trace & 8:
            with open("tmp_code.py", "w") as file:
                file.write(code)

        return code

    def extract_hand_data(self, res, hand_idx):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx]
        hand.rect_points = mpu.rotated_rect_to_points(
            hand.rect_x_center_a,
            hand.rect_y_center_a,
            hand.rect_w_a,
            hand.rect_h_a,
            hand.rotation,
        )
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res["rrn_lms"][hand_idx]).reshape(-1, 3)
        hand.landmarks = (
            (np.array(res["sqn_lms"][hand_idx]) * self.frame_size)
            .reshape(-1, 2)
            .astype(np.int32)
        )
        if self.xyz:
            hand.xyz = np.array(res["xyz"][hand_idx])
            hand.xyz_zone = res["xyz_zone"][hand_idx]
        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and
        # from rect_points
        if self.pad_h > 0:
            hand.landmarks[:, 1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:, 0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        # World landmarks
        if self.use_world_landmarks:
            hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)

        if self.use_gesture:
            mpu.recognize_gesture(hand)

        return hand

    def next_frame(self):
        self.fps.update()

        if self.laconic:
            video_frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            in_video: ImgFrame = cast(ImgFrame, self.q_video.get())
            video_frame = in_video.getCvFrame()

            # For debugging
        if self.trace & 4:
            pre_body_manip: ImgFrame = cast(ImgFrame, self.q_pre_body_manip_out.tryGet())
            if pre_body_manip:
                pre_pd_manip = pre_body_manip.getCvFrame()
                cv2.imshow("pre_body_manip", pre_pd_manip)
            pre_pd_manip: ImgFrame = cast(ImgFrame, self.q_pre_pd_manip_out.tryGet())
            if pre_pd_manip:
                pre_pd_manip = pre_pd_manip.getCvFrame()
                cv2.imshow("pre_pd_manip", pre_pd_manip)
            pre_lm_manip: ImgFrame = cast(ImgFrame, self.q_pre_lm_manip_out.tryGet())
            if pre_lm_manip:
                pre_lm_manip = pre_lm_manip.getCvFrame()
                cv2.imshow("pre_lm_manip", pre_lm_manip)

        # Get result from device
        res = marshal.loads(cast(ImgFrame, self.q_manager_out.get()).getData())
        hands = []
        for i in range(len(res.get("lm_score", []))):
            hand = self.extract_hand_data(res, i)
            hands.append(hand)

        # Statistics
        if self.stats:
            if res["bd_pd_inf"] == 1:
                self.nb_frames_body_inference += 1
            elif res["bd_pd_inf"] == 2:
                self.nb_frames_body_inference += 1
                self.nb_frames_pd_inference += 1
            else:
                if res["nb_lm_inf"] > 0:
                    self.nb_frames_lm_inference_after_landmarks_ROI += 1
            if res["nb_lm_inf"] == 0:
                self.nb_frames_no_hand += 1
            else:
                self.nb_frames_lm_inference += 1
                self.nb_lm_inferences += res["nb_lm_inf"]
                self.nb_failed_lm_inferences += res["nb_lm_inf"] - len(hands)

        return video_frame, hands, None

    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            nb_frames = self.fps.nb_frames()
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {nb_frames})")
            print(
                "# frames w/ no hand           : "
                f"{self.nb_frames_no_hand} ({100 * self.nb_frames_no_hand / nb_frames:.1f}%)"
            )
            print(
                "# frames w/ body detection    : "
                f"{self.nb_frames_body_inference} ({100 * self.nb_frames_body_inference / nb_frames:.1f}%)"
            )
            print(
                "# frames w/ palm detection    : "
                f"{self.nb_frames_pd_inference} ({100 * self.nb_frames_pd_inference / nb_frames:.1f}%)"
            )
            post_palm_frames = self.nb_frames_lm_inference - self.nb_frames_lm_inference_after_landmarks_ROI
            print(
                "# frames w/ landmark inference : "
                f"{self.nb_frames_lm_inference} ({100 * self.nb_frames_lm_inference / nb_frames:.1f}%)- "
                f"# after palm detection: {post_palm_frames} - "
                f"# after landmarks ROI prediction: {self.nb_frames_lm_inference_after_landmarks_ROI}"
            )
            if self.nb_lm_inferences:
                print(
                    f"# lm inferences: {self.nb_lm_inferences} - "
                    f"# failed lm inferences: {self.nb_failed_lm_inferences} ("
                    f"{100 * self.nb_failed_lm_inferences / self.nb_lm_inferences:.1f}%)"
                )
