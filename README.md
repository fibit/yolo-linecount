# yolo-linecount

Подсчёт объектов по пересечению линии на видео, камере или RTSP/HTTP-потоке.

## ⚙️ Требования

Обязательно:

- Python 3.10+;
- FFmpeg.

Опционально:

- NVIDIA GPU и драйвер с поддержкой CUDA, если используете CUDA-зависимости;
- TensorRT, если хотите запускать локальную модель через `.engine`;
- Docker с доступом к GPU, если хотите запускать модель через Triton.

`requirements-cuda.txt` использует PyTorch wheels из индекса `https://download.pytorch.org/whl/cu130`. Если нужна другая версия CUDA, замените индекс на подходящий из PyTorch.

## 🛠️ Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/fibit/yolo-linecount
   cd yolo-linecount
   ```

2. Создайте виртуальное окружение:
   ```bash
   python -m venv .venv
   ```

3. Активируйте виртуальное окружение:
   ```bash
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate # Linux
   ```

4. Установите зависимости (выберите вариант):
   ```bash
   pip install -r requirements-cpu.txt  # CPU
   pip install -r requirements-cuda.txt # CUDA
   ```

4. Поместите файл модели в папку `models/` (например, `models/yolo26n.pt`).

## 🚀 Опционально: TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt/download/) нужен, если модель запускается локально на NVIDIA GPU и хочется получить больше скорости. При экспорте YOLO в `.engine` модель оптимизируется под конкретную видеокарту: инференс обычно быстрее, а нагрузка на CPU ниже.

Сначала установите Python wheel TensorRT:

```bash
pip install "/path/to/TensorRT/python/tensorrt-<version>-cp<pyver>-none-<platform>.whl"
```

Экспортируйте модель в TensorRT:

```bash
yolo export model=./models/yolo26n.pt format=engine imgsz=320 half device=0
```

После успешного экспорта появится файл:

- `models/yolo26n.engine`

## 🚀 Опционально: TritonIS

Triton нужен, когда несколько камер обрабатываются отдельными процессами `linecount.py`, но модель YOLO хочется держать в одном месте. В этом случае Triton один раз загружает модель на GPU, а все процессы отправляют ему кадры по HTTP.

Установите HTTP-клиент Triton в активное виртуальное окружение:

```bash
pip install tritonclient[http]
```

### 📦 Репозиторий моделей Triton

Команды ниже выполняются из корня проекта. Создайте структуру для модели `yolo26n`:

```text
models/triton/
  yolo26n/
    1/
      model.plan
    config.pbtxt
```

Экспортируйте YOLO в ONNX:

```shell
yolo export model=./models/yolo26n.pt format=onnx imgsz=320 simplify=True
```

Соберите TensorRT plan внутри Docker-образа Triton:

```shell
docker run --rm --gpus all -it -v "${PWD}:/work" -w /work nvcr.io/nvidia/tritonserver:26.02-py3 bash -lc "trtexec --onnx=/work/models/yolo26n.onnx --saveEngine=/work/models/triton/yolo26n/1/model.plan --fp16"
```

Создайте `models/triton/yolo26n/config.pbtxt`:

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

Запуск Triton:

```shell
docker run --rm --gpus all -p 8000:8000 -p 8002:8002 -v "${PWD}/models/triton:/models" nvcr.io/nvidia/tritonserver:26.02-py3 tritonserver --model-repository=/models
```

В `config.json` укажите Triton endpoint:

```json
"model": "http://127.0.0.1:8000/yolo26n"
```

Значение `resize` должно совпадать с `imgsz`, под которое экспортировали модель.

## 🛠️ Настройка `config.json`

Скопируйте пример:

```bash
cp config.json.example config.json
```

Параметры конфигурации:

- `model` — имя файла из `models/` (например, `yolo26n.pt`) или Triton (`http://127.0.0.1:8000/yolo26n`);
- `source` — источник видео:
  - RTSP/HTTP URL (`rtsp://...`, `http://...`);
  - индекс или имя камеры (например, `0` или `"USB CAMERA"`);
  - путь к видеофайлу;
- `resize` — размер кадра для инференса (например, `320`);
- `line_start` — стартовая точка линии `[x, y]`;
- `line_end` — конечная точка линии `[x, y]`;
- `classes` — список классов для детекции (например, `[0]` для людей); `[]` или `null` — все классы модели;
- `output` — имя подпапки в `outputs/`, куда сохраняются результаты.

## ▶️ Запуск

Базовый запуск:

```bash
python linecount.py --config config.json
```

С предпросмотром видео:

```bash
python linecount.py --config config.json --preview
```

В режиме предпросмотра для остановки нажмите `q` или `Esc`.

## 📊 Результаты в `outputs/<output>/`

- `YYYY-MM-DD.jsonl` — события пересечения линии, по одному JSON-объекту в строке;
- `last_original.jpg` / `last_annotated.jpg` — периодические кадры;
- `last_original_in.jpg` / `last_annotated_in.jpg` — последний кадр при входе;
- `last_original_out.jpg` / `last_annotated_out.jpg` — последний кадр при выходе.

Пример строки события:

```json
{"timestamp": "2026-04-23T12:00:00.123", "direction": "in", "track_id": 7}
```
