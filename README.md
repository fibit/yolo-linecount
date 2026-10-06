# yolo-linecount

Count objects by line crossing on a video, camera, or RTSP/HTTP stream.

## ⚙️ Requirements

Required:

- Python 3.10+;
- FFmpeg.

Optional:

- NVIDIA GPU and a CUDA-enabled driver, if you use the CUDA dependencies;
- TensorRT, if you want to run a local model via `.engine`;
- Docker with GPU access, if you want to run the model via Triton.

`requirements-cuda.txt` uses PyTorch with CUDA 13.0. For a different CUDA version, replace the PyTorch index. Dependency versions are pinned — update them in `requirements-*.txt`.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fibit/yolo-linecount
   cd yolo-linecount
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```bash
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate # Linux
   ```

4. Install the dependencies (choose one option):
   ```bash
   pip install -r requirements-cpu.txt  # CPU
   pip install -r requirements-cuda.txt # CUDA
   ```

5. Put the model file into the `models/` folder (for example, `models/yolo26n.pt`).

## 🚀 Optional: TensorRT

[TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html) is needed if the model runs locally on an NVIDIA GPU and you want more speed. When you export YOLO to `.engine`, the model is optimized for a specific graphics card: inference is usually faster and CPU load is lower.

Install TensorRT from the NVIDIA index (the CUDA 13.x build is installed by default; for CUDA 12.x use `tensorrt-cu12`):

```bash
pip install tensorrt --extra-index-url https://pypi.nvidia.com
```

Export the model to TensorRT (`quantize=16` — FP16, the `half` argument is deprecated):

```bash
yolo export model=./models/yolo26n.pt format=engine imgsz=320 quantize=16 device=0
```

After a successful export, the following files appear:

- `models/yolo26n.onnx` — the intermediate ONNX;
- `models/yolo26n.engine` — the finished model.

## 🚀 Optional: TritonIS

Triton is needed when multiple cameras are processed by separate `linecount.py` processes, but you want to keep the YOLO model in one place. In this case, Triton loads the model onto the GPU once, and all processes send frames to it over HTTP.

Install the Triton HTTP client into the active virtual environment:

```bash
pip install tritonclient[http]
```

### 📦 Triton model repository

The commands below are run from the project root. Create the structure for the `yolo26n` model:

```text
models/triton/
  yolo26n/
    1/
      model.plan
    config.pbtxt
```

Export YOLO to ONNX:

```shell
yolo export model=./models/yolo26n.pt format=onnx imgsz=320 simplify=True
```

Build the TensorRT plan inside the Triton Docker image:

```shell
docker run --rm --gpus all -it -v "${PWD}:/work" -w /work nvcr.io/nvidia/tritonserver:26.02-py3 bash -lc "trtexec --onnx=/work/models/yolo26n.onnx --saveEngine=/work/models/triton/yolo26n/1/model.plan --fp16"
```

Create `models/triton/yolo26n/config.pbtxt`:

```protobuf
name: "yolo26n"
platform: "tensorrt_plan"
max_batch_size: 1

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 320, 320 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 300, 6 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

Starting Triton:

```shell
docker run --rm --gpus all -p 8000:8000 -p 8002:8002 -v "${PWD}/models/triton:/models" nvcr.io/nvidia/tritonserver:26.02-py3 tritonserver --model-repository=/models
```

In `config.json`, specify the Triton endpoint:

```json
"model": "http://127.0.0.1:8000/yolo26n"
```

The `resize` value must match the `imgsz` the model was exported for.

## 🛠️ Configuring `config.json`

Copy the example:

```bash
cp config.json.example config.json
```

Configuration parameters:

- `model` — the file name from `models/` (for example, `yolo26n.pt`) or Triton (`http://127.0.0.1:8000/yolo26n`);
- `source` — the video source:
  - RTSP/HTTP URL (`rtsp://...`, `http://...`);
  - camera index or name (for example, `0` or `"USB CAMERA"`);
  - path to a video file;
- `resize` — the frame size for inference (for example, `320`);
- `line_start` — the line's start point `[x, y]`;
- `line_end` — the line's end point `[x, y]`;
- `classes` — the classes to detect (for example, `[0]`); `[]` or `null` — all model classes;
- `output` — the name of the subfolder in `outputs/` where the results are saved.

You can draw the line and get the coordinates for the config in [PolygonZone](https://polygonzone.roboflow.com/).

## 🛠️ `.env` environment variables

Tracker tuning and capture behavior are configured via `.env` (copy it from `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `LINECOUNT_MODELS_DIR` | `models` | Folder with model files |
| `LINECOUNT_OUTPUTS_DIR` | `outputs` | Folder with results |
| `LINECOUNT_SNAP_EVERY` | `60` | Save frames every N processed frames |
| `LINECOUNT_READ_TIMEOUT` | `8.0` | Seconds to wait for a frame from the network before reconnecting |
| `LINECOUNT_RECONNECT_BACKOFF_MAX` | `30.0` | Maximum reconnect pause in seconds |
| `LINECOUNT_FFMPEG_STOP_TIMEOUT` | `10.0` | Seconds to wait for FFmpeg to stop |
| `LINECOUNT_NET_PREFIXES` | `rtsp://,rtmp://,...` | Network source prefixes (comma-separated) |
| `BYTETRACK_LOST_TRACK_BUFFER` | `60` | How many frames to keep lost tracks |
| `BYTETRACK_FRAME_RATE` | `30.0` | Expected FPS for the tracker |
| `BYTETRACK_TRACK_ACTIVATION_THRESHOLD` | `0.4` | Track activation threshold |
| `BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES` | `2` | Frames before a track is confirmed |
| `BYTETRACK_MINIMUM_IOU_THRESHOLD` | `0.1` | Minimum IoU for association |
| `BYTETRACK_HIGH_CONF_DET_THRESHOLD` | `0.4` | High-confidence detection threshold |
| `LINEZONE_MINIMUM_CROSSING_THRESHOLD` | `2` | Frames in the zone to confirm a crossing |

## ▶️ Running

Basic launch:

```bash
python linecount.py --config config.json
```

With video preview:

```bash
python linecount.py --config config.json --preview
```

In preview mode, press `q` or `Esc` to stop.

## 📊 Results in `outputs/<output>/`

- `YYYY-MM-DD.jsonl` — line crossing events, one JSON object per line;
- `last_original.jpg` / `last_annotated.jpg` — periodic frames;
- `last_original_in.jpg` / `last_annotated_in.jpg` — the last frame on entry;
- `last_original_out.jpg` / `last_annotated_out.jpg` — the last frame on exit.

Example event line:

```json
{"timestamp": "2026-04-23T12:00:00.123", "direction": "in", "track_id": 7}
```
