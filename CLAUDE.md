# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hand tracking system using Google Mediapipe models on Luxonis DepthAI hardware (OAK-D, OAK-D lite, OAK-1). Implements a 2-stage pipeline: palm detection followed by landmark regression. Supports gesture recognition, spatial location (XYZ), and 2D/3D interaction modes.

## Installation

### Prerequisites

On macOS:
```bash
# Install tkinter support for Python 3.11
brew install python-tk@3.11

# Install Mesa for 3D rendering (offscreen/EGL support)
brew install mesa
```

### Setup Virtual Environment

```bash
# Create virtual environment with Python 3.11 (required for TensorFlow 2.13)
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Some optional packages (`fbx`, `pythonOCC`) are not available on PyPI and can be skipped for core functionality.

**Important:** This codebase requires DepthAI SDK v2.x. The requirements.txt pins `depthai<3.0` because v3 has breaking API changes that require code updates. See [v2 vs v3 migration guide](https://docs.luxonis.com/software-v3/depthai/tutorials/v2-vs-v3/).

### Hardware Requirements

- **With DepthAI hardware:** All features available (depth, spatial location, edge mode)
- **Without hardware (webcam only):** Use Host mode with `-i 0` flag (no edge mode, no depth features)

## Running the Application

### Basic Demo Commands

**With DepthAI hardware:**
- **Host mode (default):** `./demo.py` or `./demo_bpf.py`
- **Edge mode (recommended for internal camera):** `./demo.py -e` or `./demo_bpf.py -e`
- **With gesture recognition:** `./demo.py -e -g`
- **With Body Pre Focusing:** `./demo_bpf.py -e -g -bpf higher`
- **Spatial location (depth-capable devices):** `./demo.py -e -xyz`

**Without DepthAI hardware (using webcam):**
- **With webcam:** `./demo.py -i 0` (use Host mode only, `-e` flag will fail)
- **With video file:** `./demo.py -i path/to/video.mp4`

### Controller Application

Run the GUI controller:
```bash
source venv/bin/activate
python controller.py
```

The controller GUI includes a "Use DepthAI Hardware" checkbox:
- **Checked (default):** Requires DepthAI hardware, uses edge mode for optimal performance
- **Unchecked:** Fallback to webcam mode for testing without hardware

**Interaction Modes:**
- **Draw Mode:** Draw in air with index finger
- **2D Interaction:** Load PNG/JPG images and interact with them via hand gestures
- **3D Interaction:** Load STL/OBJ 3D models, rotate, resize, and position them with hand gestures
  - Requires Mesa for offscreen rendering on macOS: `brew install mesa`
  - Uses EGL platform for rendering without GUI window (avoids macOS threading restrictions)

Use `demo.py` for basic hand tracking without Body Pre Focusing, or `demo_bpf.py` when you need Body Pre Focusing (recommended when person is >1.5m from camera).

### Key Arguments

- `-e/--edge`: Edge mode (device-side processing, faster with internal camera)
- `-s/--solo`: Solo mode (1 hand max, faster than Duo mode)
- `-g/--gesture`: Enable gesture recognition
- `-bpf/--body_pre_focusing {right,left,group,higher}`: Enable Body Pre Focusing for long-distance detection
- `-ah/--all_hands`: Consider all hands, not just raised hands
- `--lm_model {full,lite,sparse}`: Choose landmark model version
- `-xyz`: Enable spatial location measurement
- `-c/--crop`: Center crop to square
- `-r/--resolution {full,ultra}`: Sensor resolution

## Architecture

### Core Components

1. **HandTracker** (4 implementations):
   - `HandTracker.py`: Host mode, no Body Pre Focusing
   - `HandTrackerEdge.py`: Edge mode, no Body Pre Focusing
   - `HandTrackerBpf.py`: Host mode, with Body Pre Focusing
   - `HandTrackerBpfEdge.py`: Edge mode, with Body Pre Focusing

2. **HandTrackerRenderer** (`HandTrackerRenderer.py`): Rendering and visualization

3. **Pipeline Management**: Template-based scripting node code generation
   - `template_manager_script_solo.py`: Solo mode pipeline template
   - `template_manager_script_bpf_solo.py`: Solo mode with BPF template
   - `template_manager_script_duo.py`: Duo mode pipeline template
   - `template_manager_script_bpf_duo.py`: Duo mode with BPF template

4. **Utilities**:
   - `mediapipe_utils.py`: Core data structures (`HandRegion`, `HandednessAverage`), anchor generation, NMS
   - `controller.py`: GUI controller for interaction modes
   - `FPS.py`: FPS tracking utilities

### Data Structures

**HandRegion** (`mediapipe_utils.py:11`): Stores all detected hand information
- Detection data: `pd_score`, `pd_box`, `pd_kps`
- Bounding box: `rect_x_center`, `rect_y_center`, `rect_w`, `rect_h`, `rotation`
- Landmarks: `norm_landmarks` (normalized 3D), `landmarks` (2D pixels), `world_landmarks` (3D meters)
- Classification: `handedness` (0-1 float), `label` ("left"/"right")
- Optional: `gesture`, `xyz`, `xyz_zone`

### Modes

**Solo vs Duo Mode:**
- Solo: Detects 1 hand max, runs palm detection only when needed (faster)
- Duo: Detects 2 hands max (one per handedness), has `single_hand_tolerance_thresh` parameter

**Host vs Edge Mode:**
- Host: Processing on host CPU, works with external inputs (video/image files, webcam)
- Edge: Processing on device, works only with internal camera (faster, minimal data transfer)

**Body Pre Focusing (BPF):**
- Uses Movenet body pose estimation to locate hands before palm detection
- Recommended when person is >1.5m from camera
- Options: `right`, `left`, `group`, `higher`
- Can filter to only raised hands with `hands_up_only` mode

### Pipeline Flow

1. **Palm Detection**: Runs on first frame or when tracking is lost
2. **Landmark Regression**: Runs on detected hand ROI, computes next frame ROI from landmarks
3. **Gesture Recognition** (optional): Recognizes ONE, TWO, THREE, FOUR, FIVE, FIST, OK, PEACE
4. **Spatial Location** (optional): Measures XYZ coordinates of wrist/palm center

### Models

Located in `models/` directory:
- `palm_detection_sh4.blob`: Palm detection model (Mediapipe 0.8.0)
- `hand_landmark_full_sh4.blob`: Full landmark model (most accurate, slower)
- `hand_landmark_lite_sh4.blob`: Lite landmark model (faster, less accurate)
- `hand_landmark_sparse_sh4.blob`: Sparse landmark model (~10% faster than full)
- `hand_landmark_080_sh4.blob`: Legacy model (0.8.0, fastest)

Body pose models (for BPF):
- `movenet_singlepose_lightning_sh4.blob`
- `movenet_singlepose_thunder_sh4.blob`

Custom model:
- `custom_models/PDPostProcessing_top2_sh1.blob`: Edge mode palm detection post-processing

### Model Conversion

Models are converted from TFLite → TensorFlow → OpenVINO IR → MyriadX blob using PINTO's tflite2tensorflow tool.

To regenerate models:
1. Start docker container: `./docker_tflite2tensorflow.sh`
2. From container: `cd models && ./convert_models.sh`
3. For custom SHAVE count: `./gen_blob_shave.sh -m model.xml -n <shave_count>`

Custom post-processing model:
1. `cd custom_models`
2. Generate ONNX: `python generate_postproc_onnx.py`
3. Convert to blob: `./convert_model.sh`

## Development Patterns

### Creating a HandTracker Instance

```python
from HandTrackerEdge import HandTracker
from HandTrackerRenderer import HandTrackerRenderer

tracker = HandTracker(
    input_src="rgb",  # or video path, or "rgb_laconic" for no frame transfer
    use_lm=True,
    lm_model='lite',  # 'full', 'lite', 'sparse', or blob path
    solo=False,  # True for 1 hand max
    use_gesture=True,
    xyz=False,  # Enable spatial location
)

renderer = HandTrackerRenderer(tracker=tracker)

while True:
    frame, hands, bag = tracker.next_frame()
    if frame is None: break
    frame = renderer.draw(frame, hands, bag)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

renderer.exit()
tracker.exit()
```

### Landmark Access

Hand landmarks follow Mediapipe's 21-point hand model (indices 0-20):
- 0: WRIST
- 1-4: THUMB (CMC, MCP, IP, TIP)
- 5-8: INDEX (MCP, PIP, DIP, TIP)
- 9-12: MIDDLE (MCP, PIP, DIP, TIP)
- 13-16: RING (MCP, PIP, DIP, TIP)
- 17-20: PINKY (MCP, PIP, DIP, TIP)

Access via:
- `hand.landmarks`: 2D pixel coordinates in source image
- `hand.norm_landmarks`: 3D normalized [0,1] coordinates
- `hand.world_landmarks`: 3D meter coordinates (if `use_world_landmarks=True`)

### Template Script Substitution

Edge mode uses template-based script generation. Variables like `${_pad_h}`, `${_img_h}`, `${_frame_size}` are substituted at runtime in `HandTrackerEdge.py` or `HandTrackerBpfEdge.py`.

Key substitution variables:
- `${_TRACE1}`, `${_TRACE2}`: Debug tracing
- `${_pad_h}`, `${_img_h}`, `${_img_w}`, `${_frame_size}`, `${_crop_w}`: Image dimensions
- `${_pd_score_thresh}`, `${_lm_score_thresh}`: Score thresholds
- `${_body_pre_focusing}`, `${_body_score_thresh}`, `${_hands_up_only}`: BPF parameters
- `${_IF_USE_HANDEDNESS_AVERAGE}`: Conditional code block markers

## Performance Considerations

- **FPS varies significantly**: Faster when hands are detected (landmark model only) vs no hands (palm detection runs every frame)
- **Tune `internal_fps`**: Start with default, adjust based on observed FPS
- **Edge mode > Host mode** when using internal camera
- **Solo mode > Duo mode** for single hand tracking
- **BPF adds overhead** during palm detection phase, but no overhead once hand is tracked
- In Duo mode with 1 hand: `single_hand_tolerance_thresh` controls palm detection frequency (lower = more responsive, higher = faster when second hand rarely appears)

## Controller Implementation

### Cross-Platform Design

The controller (`controller.py`) is designed to work on Windows, macOS, and Linux:
- Uses `sys.executable` to launch demo.py with the current Python interpreter
- Uses `os.path.join()` for cross-platform file paths
- Automatically detects and adapts to the operating system

### Hardware Mode Toggle

The controller includes a "Use DepthAI Hardware" checkbox that configures demo.py launch parameters:
- **DepthAI mode (checked):** Adds `--edge` flag, uses device's internal camera
- **Webcam mode (unchecked):** Adds `-i 0` flag, uses system webcam

### 3D Rendering Implementation

For 3D interaction mode (STL/OBJ files), the controller sets `PYOPENGL_PLATFORM=egl` environment variable before launching demo.py:
- Uses EGL (OpenGL offscreen rendering) instead of creating GUI windows
- Avoids macOS threading restrictions ("API misuse: setting the main menu on a non-main thread" errors)
- Renders 3D models to memory buffers that are composited into the camera feed
- Supports rotation, scaling, and positioning via hand gestures

### Socket Communication

`controller.py` communicates with `demo.py` via socket (default port 54465):
- `interact2D <file_path>`: Enable 2D interaction mode
- `interact3D <file_path>`: Enable 3D interaction mode
- `draw`: Enable drawing mode
- `hide`: Hide extras, disable FPS
- `show`: Show extras, enable FPS

The controller waits up to 5 seconds for demo.py to start the socket server, with 3 retry attempts.

## Virtual Camera Support

The application can output to a virtual camera for use in video conferencing apps (Zoom, Meet, etc.):

### Setup
- **macOS/Windows:** Install OBS and run it once to register the virtual camera
- **Linux:** Install v4l2loopback: `sudo modprobe v4l2loopback`

### Usage
- Enable via `--virtual_cam` flag or "Virtual Camera" checkbox in controller.py
- Virtual camera appears as "OBS Virtual Camera" in video apps
- Frame is NOT mirrored in virtual camera mode (text/UI elements are flipped to appear correctly)

### Implementation (`HandTrackerRenderer.py`)
- Uses `pyvirtualcam` library with OBS backend
- Converts BGR→RGB before sending frames
- Uses `sleep_until_next_frame()` for frame pacing
- Proper cleanup on exit via `virtual_cam_output.close()`

## In-Stream Pie Menu

A fist-activated radial menu allows toggling features without leaving the camera view:

### Usage
1. Make a fist and hold for 1 second (progress circle appears)
2. Pie menu appears with options: Draw, Extras (show/hide)
3. Move fist to an icon and hold 0.5s to activate, OR move past the icon for quick-select
4. Selected icon shows updated state with fade animation
5. Release fist to close menu; menu won't reopen until a new fist is made

### Implementation (`HandTrackerRenderer.py`)
- `_handle_pie_menu()`: Main handler for fist detection and menu logic
- `_draw_pie_icon()`: Renders individual menu icons
- `_draw_fading_icon()`: Handles post-selection fade animation
- `_get_fist_center()`: Calculates palm center using landmarks 0, 5, 9, 17
- `_draw_text()`: Renders text correctly in both mirrored and non-mirrored modes

## Draw Mode

### Pinch-to-Draw Gesture
- Pinch thumb and index finger together (distance < 40px threshold)
- Drawing point is the midpoint between thumb tip (landmark 4) and index tip (landmark 8)
- Hold pinch for 0.05s before drawing starts
- PEACE gesture clears all drawn lines (hold 0.5s)
- FOUR gesture acts as eraser (30px radius)

## Model Optimization

Models are compiled with different SHAVE counts for the Myriad X VPU:
- **6-shave models** (default): Faster inference, recommended for 2-3 concurrent models
- **4-shave models**: Lower resource usage, for running many models in parallel

The codebase uses 6-shave models where available:
- `palm_detection-2021-02-27_sh6.blob`
- `hand_landmark_full-2022-11-10_sh6.blob`
- `hand_landmark_lite-2022-11-12_sh6.blob`
- `hand_landmark_sparse_sh4.blob` (no sh6 version available)
