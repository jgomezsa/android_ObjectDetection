# Technical documentation

Everything a developer needs to build, understand, measure and change the app. The [README](README.md) is the user-facing introduction.

## 1. Overview

An Android app (Kotlin, Jetpack Compose) that runs YOLOX object detection on live camera frames, entirely on the device, and draws boxes with labels over the
analysed camera picture. The user picks one of two models (YOLOX-S or YOLOX-Tiny). The model runs on the phone's **GPU through LiteRT**, and falls back to the **CPU**
(also LiteRT) when the GPU cannot run it. The camera is read with **Camera2**, which can reach the ultrawide lens on phones that hide it from CameraX.
The user can add more YOLOX models (`.tflite`, or `.onnx` which runs on the CPU through ONNX Runtime) from a web address, and delete them again (5.8).

- Package `com.example.objectdetection`, single module `:app`, entry point `MainActivity.kt`
- Remote: https://github.com/jgomezsa/android_ObjectDetection
- Tested on a Poco X7 Pro (Dimensity 8400-Ultra, Mali GPU, MediaTek APU, Android 15 / HyperOS)

## 2. Stack and toolchain

| Item | Version / note |
|---|---|
| Kotlin | 2.0.21 (`-Xskip-metadata-version-check` is set because LiteRT 2.3.0 is built with a newer Kotlin) |
| Android Gradle Plugin / Gradle | 8.7.3 / 8.9 (wrapper, committed) |
| JDK | 17 (Microsoft OpenJDK), `JAVA_HOME` set as a user env var |
| compileSdk / targetSdk / minSdk | 35 / 35 / 26 (adaptive icons need no PNG fallbacks) |
| UI | Jetpack Compose, Material 3, BOM 2024.10.01 |
| Camera | Camera2 (platform API, no library) |
| Inference | LiteRT `com.google.ai.edge.litert:litert` and `litert-gpu` 2.3.0, `CompiledModel` with the GPU or CPU accelerator |
| ONNX | ONNX Runtime Android 1.20.0 (`onnxruntime-android`), CPU only, for downloaded `.onnx` models |
| Permissions | `CAMERA`; `INTERNET`, used only by the downloads of "Manage models" (https only) |
| Unit tests | JUnit 4 in `app/src/test/` (host JVM, no device) |

Versions live in the Gradle Kotlin DSL files. Android Studio-only files are not used (`.idea/` is ignored).

## 3. Development environment (Windows 11)

- Android SDK at `%LOCALAPPDATA%\Android\Sdk` (`ANDROID_HOME`), installed with the command-line tools only: platform-tools, `platforms;android-35`,
  `build-tools;35.0.0`, emulator, `system-images;android-35;google_apis;x86_64`.
- User PATH includes `platform-tools`, `cmdline-tools\latest\bin`, `emulator` (open a new terminal after changes).
- `local.properties` (git-ignored) holds `sdk.dir`; regenerate it if the SDK moves.
- AVD `Pixel_API35` (Pixel 7, API 35, x86_64). The emulator needs CPU virtualization (BIOS + Windows Hypervisor Platform);
  `emulator -accel-check` should report "WHPX is installed and usable". The emulator camera only shows a virtual scene; a real phone is needed for real tests.
- VS Code extensions: `fwcd.kotlin` (language server) and `mathiasfrohlich.Kotlin` (syntax).
- Python tooling (model export, conversion, icon) runs through `uv` (`uv run --python 3.12 --with <packages> ...`); no global Python.
- Shell is PowerShell; use `.\gradlew.bat` (not `./gradlew`).

### Running on a physical phone

1. Settings > About phone > tap "Build number" 7 times to enable Developer options.
2. Developer options > "USB debugging" (or "Wireless debugging" and `adb pair`). On Xiaomi/Poco/Redmi also enable "Install via USB" and
   "USB debugging (Security settings)", keep the phone unlocked and accept the install prompt (otherwise `INSTALL_FAILED_USER_RESTRICTED`).
3. Connect via USB, accept the RSA prompt, check `adb devices` shows `device`.
4. `.\gradlew.bat installDebug`, then launch from the phone or `adb shell am start -n com.example.objectdetection/.MainActivity`.

## 4. Commands

```powershell
.\gradlew.bat test assembleDebug         # verify after any change (unit tests + debug APK)
.\gradlew.bat installDebug               # build + install on the connected device/emulator
adb shell am start -n com.example.objectdetection/.MainActivity
adb devices                              # list phones/emulators
adb logcat -s AndroidRuntime:E           # crash logs
adb logcat -s Inference:D Camera2:D      # per-frame timing, model load, backend used, chosen camera and zoom stops
emulator -avd Pixel_API35                # start the emulator
sdkmanager --list_installed              # SDK packages
```

Add `--no-daemon` to Gradle if memory is tight. Screenshot: `adb shell screencap -p /sdcard/s.png` then `adb pull /sdcard/s.png`
(PowerShell `>` redirection corrupts binary). The screenshot is black when the phone is locked or its screen is off.

Developer overrides (no UI), to benchmark a model/backend combination:

```powershell
adb shell am start -n com.example.objectdetection/.MainActivity --es model yolox_tiny --es backend CPU
# model: yolox_s | yolox_tiny (asset name without .tflite); backend: GPU | CPU (forces a single backend, no fallback)
adb shell "run-as com.example.objectdetection sh -c 'rm -rf /data/data/com.example.objectdetection/cache/*'"   # clear the GPU program cache (cold-start test)
```

The model picked in the UI is remembered (`SharedPreferences` "settings", key `model`); the `model` override is not saved.

## 5. Architecture

### 5.1 Frame pipeline

```
Camera2 ImageReader (YUV_420_888, 3 buffers, acquireLatestImage, 4:3 around 1280x960)
  -> YuvConverter: YUV -> ARGB (BT.601 full range) and rotation to upright in one pass, rows split over the CPU cores -> Bitmap
  -> LetterboxRenderer: scale to fit a square (gray 114 padding) -> ARGB pixels
  -> NHWC float tensor, BGR 0..255 (YOLOX is not normalised)
  -> model -> [1, anchors, 85]  (raw x, y, w, h + objectness + 80 class scores per anchor; anchors = 8400 at 640, 3549 at 416)
  -> YoloxPostProcessor.decode: grid/stride decode, score = objectness x class, > 0.25, class-aware NMS (IoU 0.45), map back to normalized source coordinates
  -> ShownFrame(bitmap, detections) published to Compose state; FrameView draws that same bitmap letterboxed on a Canvas and the boxes on top
```

The picture on screen is the analysed frame itself: drawing the bitmap together with its own detections avoids boxes trailing moving objects (an earlier version showed
the live preview with boxes from an older frame, 100-150 ms behind). The cost is that the picture updates at the detection rate (about 10-25 fps depending on the model),
and the picture is about one frame time old. The analysis size (`ANALYSIS_WIDTH` x `ANALYSIS_HEIGHT` = 1280x960 in `Camera2Source.kt`, closest supported 4:3 size) is
also what is recorded; at 640x480 the picture was visibly blurry once scaled to the screen.

Threads: Camera2 callbacks run on a "camera-control" thread; the `ImageReader` listener, the conversion, the detection and the recording all run on a separate
"camera-image" thread, one frame at a time. While a frame is processed (or a model loads) newer frames replace each other in the `ImageReader`; only the latest is taken next.
Results are published to Compose state.

### 5.2 Components (`app/src/main/java/com/example/objectdetection/`)

| File | Role |
|---|---|
| `MainActivity.kt` | Edge-to-edge host (`enableEdgeToEdge()`, `Scaffold` with `innerPadding` on the root), CAMERA permission gate, reads the `model` / `backend` launch extras |
| `CameraScreen.kt` | Screen state and layout, analyzer callback (model loading with backend fallback, detection, recording), `ModelSelector`, `FrameTimePlot`, `ZoomSelector`, buttons, `GpuBadge`, `FrameView` |
| `Camera2Source.kt` | Camera2: camera discovery (including hidden ids), session with an `ImageReader`, zoom, rotation, frame delivery |
| `CameraPicker.kt` | Orders candidate cameras: the one that zooms out the furthest first (pure Kotlin, unit tested) |
| `YuvConverter.kt` | YUV_420_888 to rotated ARGB, parallel over row bands (pure Kotlin, unit tested) |
| `BackendChain.kt` | `Backend` (GPU, CPU) and `BackendChain.firstWorking`: try backends in order until one runs the model (unit tested) |
| `LiteRtDetector.kt` | LiteRT `CompiledModel` of a `.tflite` (asset or downloaded file) on a given backend: GPU (fp16, program cache) or CPU (XNNPACK threads); self-test on creation; input size and class count derived from the tensors |
| `OrtDetector.kt` | ONNX Runtime session of a downloaded `.onnx` on the CPU (NCHW float input, direct `FloatBuffer`); shapes checked with `ModelContract`, self-test on creation |
| `Models.kt` | `ModelFormat` (TFLITE, ONNX) with detection from the first bytes, `ModelInfo`, `BuiltInModels`, `DEFAULT_MODEL`, `ModelRejectedException` (pure Kotlin, unit tested) |
| `ModelSpec.kt` | `OutputLayout`, `InputConvention` and `ModelSpec`: what kind of model it is and how to feed and decode it (pure Kotlin) |
| `ModelContract.kt` | Recognises the model type from a test run and rejects unsupported ones with a reason (pure Kotlin, unit tested) |
| `OutputDecoders.kt` | Decoders for every layout and `InputPacker` (pure Kotlin, unit tested) |
| `ModelStore.kt` | `ModelStore` (downloaded models on disk), `LabelParser`, `ModelUrl` (pure Kotlin, unit tested) |
| `ModelDownloader.kt` | https download with redirects, size cap, cancel, plain-language errors (java.net, unit tested against a local socket server) |
| `ModelInstaller.kt` | Download, detect format, probe on the CPU, check labels, install (needs a `Context`) |
| `ManageModelsDialog.kt` | The "Manage models" overlay: list, delete with confirmation, add from addresses, progress/result, format notes (`MODEL_FORMAT_NOTES`) |
| `ObjectDetector.kt` | `ObjectDetector` interface, `DetectionResult` (with phase timings), `LetterboxRenderer` |
| `YoloxPostProcessor.kt` | Grid/stride decode of YOLOX's raw output, then NMS (pure Kotlin, unit tested) |
| `YoloPostProcessor.kt` | `Detection`, `Letterbox`, box mapping, `nms`, `iou` shared by decoders (pure Kotlin, unit tested) |
| `ZoomPresets.kt` | Zoom stops from the camera's range, selector position/ratio mapping, labels (pure Kotlin, unit tested) |
| `FrameTimeHistory.kt` | Ring buffer of the last 50 frame times plus `niceMax` (auto-scaled top tick of the plot; pure Kotlin, unit tested) |
| `DetectionRenderer.kt` | Draws boxes and labels on a native `Canvas` (sizes scale with the frame width); used by the on-screen view and by the recorder |
| `VideoRecorder.kt` | Records analysed frames plus boxes to an H.264 MP4 (`MediaCodec` surface input, `MediaMuxer`, saved through `MediaStore`) |
| `RecordingFormat.kt` | Pure helpers for recording: even sizes, bit rate, file name, m:ss time (unit tested) |
| `CocoLabels.kt` | The 80 COCO class names |

### 5.3 Backends and fallbacks

The policy for every model, now and from here on: **try the GPU first, fall back to the CPU if the model does not run on the GPU.** The user chooses the model, so a
slow load (for example the one-time GPU compilation) is acceptable; there is no load-time limit.

`createDetector` (`CameraScreen.kt`) calls `BackendChain.firstWorking(BackendChain.DEFAULT_ORDER)` = `[GPU, CPU]`, creating a `LiteRtDetector(context, "<model>.tflite", backend)` for each
backend until one works. A backend "works" when `CompiledModel.create` succeeds **and** a self-test inference (one run on a gray image) completes with the expected output size and
only finite values, so a delegate that compiles but cannot run (or returns NaN) is also rejected. If every backend fails the model is reported as failed: a toast appears, the
previous model is reloaded if there was one, otherwise the top-left text says "Model failed to load" (picking a model again retries).

- **ONNX models** (`ModelFormat.ONNX`) have a single backend, the CPU through ONNX Runtime (`createDetector` filters the order to `[CPU]`; forcing `--es backend GPU` on one fails with
  "No backend could run the model"). ONNX Runtime for Android has no GPU provider; NNAPI was tried earlier and compiled for 6-29 s on every start, so it is not used.
- **GPU:** `CompiledModel.GpuOptions(precision = FP16, serializationDir = cacheDir, modelCacheKey = "<name>-<asset size>", serializeProgramCache = true)`. The first start of a model
  compiles OpenCL programs; later starts load them from `cacheDir`. The key includes the asset size so a replaced model never reuses a stale cache. The OS may clear the cache.
- **CPU:** `Accelerator.CPU` with `CpuOptions(numThreads = cores, clamped to 2..6)` (XNNPACK), fp32.
- The backend actually used is logged (`loaded <model> <GPU|CPU> in N ms`) and shown by the badge (teal "GPU" or amber "No GPU").
- Adding a backend (for example NNAPI/NPU) means adding a `Backend` value, its `CompiledModel.Options` in `LiteRtDetector`, and putting it in `DEFAULT_ORDER`.
- **Possible later improvement (not implemented):** calibrate several backends the first time a model runs on a phone (time a few inferences on each) and remember the fastest.
  With only GPU and CPU the GPU wins whenever it works, so the simple ordered fallback is used for now.

### 5.4 User interface rules

- Top-left, a row: `FrameTimePlot` in the corner, then to its right the frame processing time in ms (`result.inferenceMs`, i.e. preprocess + model + decode),
  "Loading…" while a detector is being created, "Paused", or "Model failed to load".
- `FrameTimePlot` is a small line plot of the last 50 frame times (`FRAME_HISTORY`). Gray axes, two labelled ticks on the vertical axis (auto-scaled
  maximum from `FrameTimeHistory.niceMax`, and half of it), no horizontal ticks. The history is cleared when a model is (re)loaded.
- Top-right, a column: the model selector, and below it (once loaded) `GpuBadge`: a chip icon (`res/drawable/ic_chip.xml`, tinted) with "GPU" (teal) or "No GPU" (amber).
  The selector (`ModelSelector`) shows the model name with a small arrow ("YOLOX-S ▾"); tapping it opens a dropdown with every model (built-in and downloaded; the current one has a check mark), a divider, and "Manage models…" (5.8); choosing
  a model sets `selectedModel` and `requestedModel` (ids), which the analyzer notices on the next frame and reloads. The choice is saved in `SharedPreferences`
  (a saved id that no longer exists falls back to `DEFAULT_MODEL`).
  The provider/backend is not named anywhere except the GPU badge.
- Layout: a `Column` with the picture and its overlays in a weighted `Box` on top, and a fixed 120 dp control strip below it (zoom selector, then Stop/Record), so the
  controls never cover the image. The picture is letterboxed inside the Box; the plot, time, model selector and badge sit in the Box's corners.
- `ProcessingButton` ("Stop" with `ic_stop.xml` / "Start" with `ic_play.xml`) flips `processing` (UI state) and `processingFlag` (`AtomicBoolean` read by the analyzer thread).
  While paused the analyzer only shows the frame (no model run, so the GPU is idle); the picture stays live, boxes are hidden, the detector stays loaded and the plot keeps its values.
- `RecordButton` (red dot + "Record"; while recording a red square + elapsed m:ss; dimmed while paused). See 5.6.
- Zoom: the stops come from `ZoomPresets.forRange(min, max)` with the selected camera's zoom range: its minimum when below 1x, 1x, then 2x, 5x, 10x up to the maximum;
  fewer than two stops hides the control. `ZoomSelector` is a thin line with a dot and label per stop (evenly spaced, 44 dp from the screen edges so the first point is not in the
  system back-gesture zone) and a white thumb at the current ratio; tap snaps to the nearest stop, dragging slides continuously, and pinching on the picture
  (`detectTransformGestures`) multiplies the ratio. All three call `setZoom`, which clamps to the stops' range and calls `Camera2Source.setZoom`; the thumb position is interpolated
  between stops. The analysis frame follows the zoom, so boxes, picture and recorded video all show the zoomed image.
  Pinch-to-zoom is implemented but was not tested on the device (adb cannot send two-finger gestures); tap and drag were.
- Edge-to-edge with `Scaffold`; `innerPadding` is applied to the root composable.

### 5.5 Camera (Camera2)

`Camera2Source` replaced CameraX because CameraX only uses the cameras Android lists publicly. On the Poco X7 Pro `CameraManager.cameraIdList` has only id 0 (back, zoom 1-10) and
id 1 (front), but the characteristics of other ids are readable and some can be opened:

| Id | Listed | Zoom | What it is |
|---|---|---|---|
| 0 | yes | 1-10 | main camera |
| 2 | no | 1-4 | ultrawide physical camera |
| 3 | no | 0.6-10 | logical camera over physical 0 and 2 (zooming below 1x switches to the ultrawide) |
| 4 | no | 1-10 | logical camera over 0 and 2 (depth/bokeh features) |
| 5, 6 | no | - | "system only", refused to apps |

- **Discovery:** the listed ids plus ids 0-9 are probed with `getCameraCharacteristics` (unknown or system-only ids throw and are skipped). Usable cameras are back-facing,
  `BACKWARD_COMPATIBLE`, with `YUV_420_888` output; zoom range from `CONTROL_ZOOM_RATIO_RANGE` (API 30+, otherwise 1..`SCALER_AVAILABLE_MAX_DIGITAL_ZOOM`).
- **Choice:** `CameraPicker.order` sorts by lowest minimum zoom, then listed before hidden, then larger maximum zoom, and drops cameras that cannot reach 1x. Each camera is tried in
  that order until one opens and its session configures, so a phone that refuses the hidden camera ends up on the normal one. Here that selects id 3 (0.6x to 10x).
- **Session:** one `ImageReader` target (no preview surface), `TEMPLATE_PREVIEW`, continuous-picture autofocus when available, `CONTROL_ZOOM_RATIO` (or a centred `SCALER_CROP_REGION` below
  API 30). Size: the 4:3 `YUV_420_888` size closest to 1280x960.
- **Orientation:** clockwise rotation = `(SENSOR_ORIENTATION - display rotation + 360) % 360`, applied inside `YuvConverter`.
- **Conversion:** full-range BT.601 integer maths, handles planar and interleaved chroma (pixel stride 1 or 2), and writes pixels already rotated; the 8 row bands run in parallel.
- **Lifecycle:** a `LifecycleEventObserver` starts the source on `ON_START` and stops it (waiting for the image thread) on `ON_STOP`, which also ends a recording; the detector
  stays loaded. Zoom ratio is kept across restarts.
- Logcat (`Camera2` tag): every probed camera with its range, and `streaming camera <id> at <size>`.

### 5.6 Video recording

The UI toggles `recordFlag` (`AtomicBoolean`); the image thread owns the `VideoRecorder`: it creates it on the first frame after the flag is set (size = the upright
analysis frame, rounded down to even) and finishes it on the first frame after the flag clears. After each detection the same bitmap and its detections are drawn onto the
encoder's input `Surface` (`lockHardwareCanvas`, then `DetectionRenderer`), so boxes always match their frame. Frames are timestamped when drawn, so the video plays at the
real detection rate. H.264, bit rate `width * height * 4`, key frame every second, no audio, nothing but the camera image and boxes (no plot or buttons).

- Resolution is the analysis resolution (960x1280 upright), exactly what is shown on screen; about 0.45 MB per second of video.
- Android 10+: written to `Movies/ObjectDetection` through `MediaStore` (pending until finished, deleted if empty), no permission needed. Older versions: the app's own
  external Movies folder (not shown in the gallery).
- Pausing detection, stopping the app (`ON_STOP`) or an encoder error ends the recording. A toast tells the user where the video was saved.
- Logcat: `recording started WxH`, `recording finished, saved=<bool>`.

### 5.7 Manage models (download, check, delete)

"Manage models…" in the model menu opens `ManageModelsDialog`, an overlay drawn over the whole screen inside the main window (a `Box` around the screen content), not a platform
`Dialog`: on the Poco X7 Pro a `Dialog` window did not receive the keyboard inset, so its bottom buttons stayed under the keyboard, whereas the overlay with
`windowInsetsPadding(WindowInsets.safeDrawing)` fits above it. The progress or result line and the buttons ("Download and add" / "Cancel download", "Close") sit outside the scrolling
part so they are always visible; the back gesture closes the overlay (`BackHandler`).

- **List:** `ModelStore.all()` = the built-in models + the folders in `filesDir/models/<id>/`, each with `model.<tflite|onnx>`, `model.properties` (name, format, input size, class count, source
  address) and optionally `labels.txt`. Damaged or incomplete folders are ignored. Built-in models cannot be deleted; deleting a downloaded one asks for confirmation, removes its folder (ids that
  would leave `filesDir/models` are refused) and, if it was the selected model, switches to `DEFAULT_MODEL`.
- **Add** (`ModelInstaller.install`, on a background thread, cancellable):
  1. The address must be https with a host (`ModelUrl.parse`). 2. `ModelDownloader` downloads to `cacheDir` (redirects followed, 400 MB cap, 15 s connect / 30 s read timeouts); HTTP 401/403 and 404 get
  their own messages. 3. `ModelFormat.detect` reads the first bytes: TFLite is the FlatBuffer identifier `TFL3` at offset 4; ONNX is a protobuf starting with byte `0x08` (or a `.onnx` name);
  archives (zip, gzip, bzip2, 7z, tar), web pages (HTML/XML/JSON, for example a Hugging Face `/blob/` link instead of `/resolve/`) and anything else are rejected with an explanation.
  4. The optional class names file is downloaded (1 MB cap) and parsed (one name per line). 5. **Probe:** the model is loaded once on the CPU (a backend every model can use), which runs the same
  self-test as a normal load: the input must be a square 3-channel image with a side that is a multiple of 32 (`ModelContract.fromSizes` for TFLite, `fromOnnxShapes` with the real shapes for ONNX;
  dynamic ONNX sizes assume 640), the output size must be `anchors x (5 + classes)` for that input size, and on a gray test image the values must be finite, the objectness and class scores
  within 0..1 (`checkValues`) and the box columns small (above 64 means already decoded pixels). A YOLOv8-style `[1, 84, 8400]` output is recognised as channels-first. 6. The class names must match the class
  count (80 classes without a file use COCO names, other counts without a file get "class N"). 7. `ModelStore.install` moves the file, writes the metadata and makes the id and display name unique.
  Nothing is added unless every step passes, and the temporary file is always removed.
- The probe only proves that the model runs on the CPU; the GPU is tried (and the CPU used if it fails) when the model is selected, as for any `.tflite`.
- Limits: it cannot verify that a `.tflite` input is NHWC or that colours are BGR 0..255 (a model with another convention loads but detects nothing useful), nor that class names are correct.
- Verified on the Poco X7 Pro: `yolox_nano.tflite` (GPU, 21-25 ms), `yolox_plates_s.onnx` (CPU, 1 class), the Autoware traffic-light ONNX with its labels file (960x960, 3 classes), a web-page link, a
  YOLOv11 face ONNX (rejected as channels-first), an `http://` address and a 404 (rejected), deletion of an unused model and of the selected one.

### 5.7.1 Model types (YOLOX, YOLOv8/11, NMS-free YOLO, DETR)

A model is no longer assumed to be YOLOX. When it is added, `ModelContract.resolve` runs it on a gray image (first the 0..255 scale, then 0..1; the output *structure* does not depend on the picture) and `analyze` recognises the
output from its sizes, the ONNX shapes when known, and its values. The result is a `ModelSpec` (layout, input size, class count, queries, which output is which, normalised boxes) stored with the model, so
loading later only verifies the sizes. Each layout fixes the preprocessing and the decoder (`OutputDecoders`, `InputPacker`):

| Layout | Output | Colours / range | Resize | Decoding |
|---|---|---|---|---|
| `YOLOX_ROWS` | `[1, anchors, 5 + C]` raw offsets, objectness, probabilities | BGR 0..255 | letterbox | grid/stride decode, score = obj x class, NMS |
| `V8_CHANNELS_FIRST` (YOLOv8, YOLO11) | `[1, 4 + C, anchors]` cx, cy, w, h, class probabilities | RGB 0..1 | letterbox | best class, NMS; boxes in pixels, or fractions (Ultralytics TFLite) scaled by the input size |
| `V8_ROWS` | the same transposed | RGB 0..1 | letterbox | as above |
| `END2END` (YOLOv10) | `[1, 300, 6]` x1, y1, x2, y2, score, class | RGB 0..1 | letterbox | threshold only (pixels or fractions) |
| `DETR_SPLIT` (RT-DETR, D-FINE) | logits `[1, Q, C]` + boxes `[1, Q, 4]` (normalised cx, cy, w, h), either order | RGB 0..1 | stretched to the square | sigmoid of the best logit > 0.3, no NMS |
| `DETR_MERGED` | `[1, Q, 4 + C]` normalised box, probabilities | RGB 0..1 | stretched | threshold 0.3 |

Disambiguation: channels-first is tried before rows; the box values decide pixels (above 64), fractions (up to 1.5) or YOLOX raw offsets (small, with 5 + C columns); scores must be probabilities (-0.05..1.05); an
unknown-shape output is also tried as `Q x (4 + C)` for common query counts. Rejections explain why. Found on real files: Ultralytics TFLite exports have *normalised* boxes and 42000 values that also fit 300 queries x 140, so
the normalised check must come first (a `ModelContractTest` covers it).

Limits: only a single `float32` image input; the ImageNet mean/std normalisation of the original DETR or RF-DETR is not applied (the model loads but detects nothing useful); segmentation, pose and classification outputs are rejected; the
class count of an NMS-free output cannot be read from its shape (it takes 80, or the length of the names file). ONNX models with dynamic height/width are fed 640x640.
Verified on the Poco X7 Pro with the real files: YOLO11n ONNX (CPU, 145 ms run), YOLOv10n ONNX (CPU, 157 ms), RT-DETR v2 r18vd ONNX (CPU, 610 ms), a YOLO11 pothole TFLite (GPU, 30 ms), YOLOX files as before.

### 5.8 Logging

Tag `Inference`: `loaded <model> <BACKEND> in N ms` once per load, `zoom range a..b, stops [...]`, and per frame `<model> <BACKEND> T ms (pre a, run b, post c) frame F ms`
(`frame` also includes the camera conversion). Errors for a model that could not be loaded; warnings for a backend that failed (with the exception).

## 6. Models

Bundled in `app/src/main/assets/` (about 56 MB in total):

| File | Model | Input | Output |
|---|---|---|---|
| `yolox_s.tflite` (36 MB) | YOLOX-S | `[1, 640, 640, 3]` | `[1, 8400, 85]` |
| `yolox_tiny.tflite` (20 MB) | YOLOX-Tiny | `[1, 416, 416, 3]` | `[1, 3549, 85]` |

Source: the official YOLOX release ONNX files
`https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx` and `.../yolox_tiny.onnx` (code licence Apache-2.0; confirm the weights terms before shipping).

**Model contract.** Input: BGR, 0..255 float, no normalisation, letterboxed with gray 114. Output (the released ONNX is *not* decoded): per anchor `x, y` are offsets from the grid
cell, `w, h` are log sizes in strides, objectness and the 80 class scores are already sigmoided. `YoloxPostProcessor` decodes `(x + gx) * stride`, `exp(w) * stride`,
score = objectness x class score, threshold 0.25, class-aware NMS 0.45. Anchors are the grids of strides 8, 16, 32 one after the other, row by row (6400 + 1600 + 400 at 640;
2704 + 676 + 169 at 416). The input size is read from the model, so other sizes (multiples of 32) work.

### 6.1 Conversion (ONNX to TFLite for the GPU)

`onnx2tf` works on Windows (Ultralytics' own exporter is not involved):

```powershell
uv run --python 3.12 --with onnx2tf --with tensorflow --with tf_keras --with onnx --with onnxruntime --with onnxsim `
  --with onnx_graphsurgeon --with sng4onnx --with ai_edge_litert --with psutil --with protobuf `
  onnx2tf -i yolox_s.onnx -o out -tb tf_converter -b 1
# use out/yolox_s_float32.tflite as app/src/main/assets/yolox_s.tflite (same for yolox_tiny)
```

- **`-tb tf_converter -b 1` is mandatory.** The default `flatbuffer_direct` backend leaves dynamic-shape tensors and the GPU delegate fails with "Failed to compile model".
- Use the **float32** file: the GPU computes in fp16 by itself, and the float16 file has fp16 inputs/outputs.
- The TFLite input is NHWC; the output keeps `[1, anchors, 85]`.
- Check a converted model on the host against the ONNX one with `ai_edge_litert.interpreter.Interpreter`: max absolute difference 5.5e-5 (s) and 6.1e-5 (Tiny) on a test image.
  Run downloaded files only from their own empty folder, and Python with `-I`.

### 6.2 Adding or replacing a built-in model, and model formats

Built-in models: put `<id>.tflite` in `assets/` and add a `ModelInfo` to `BuiltInModels` (`Models.kt`). A model with a different output layout needs its own decoder.
Users add models at run time through "Manage models" (5.7); the same rules apply, and this is what the app tells them (`MODEL_FORMAT_NOTES`, also in the README):

- **`.tflite`** runs on the GPU or the CPU. Fixed input size and float tensors are required; int8 (quantised) files usually cannot use the GPU delegate and may fail to load. An ONNX model has to be
  converted on a PC first (6.1); the phone cannot convert models.
- **`.onnx`** runs on the CPU only (ONNX Runtime), several times slower than `.tflite` on the GPU, but needs no conversion. A dynamic input size is read as 640.
- **Supported types** (5.7.1): YOLOX, YOLOv8/YOLO11, NMS-free YOLO (YOLOv10), DETR-style (RT-DETR, D-FINE); single square float image input. Not supported: ImageNet-normalised models (original DETR, RF-DETR), SSD,
  EfficientDet, segmentation/pose/classification, models with extra inputs, archives.
- **Class names:** 80 classes = COCO names; otherwise a `.txt` with one name per line whose line count equals the class count, or "class N" labels.
- Downloaded models are the user's responsibility: licence (many YOLOv8/v11 models are AGPL-3.0), source, size (400 MB cap).

### 6.3 Candidate models to try (researched 2026-10-12)

URLs were checked with HEAD requests (HTTP 200) on that date. Licences are what each page says; check them again before shipping. "Raw YOLOX" means the output is the same undecoded
`[1, anchors, 5 + classes]` layout the app already decodes (verified by running the files on the host), so only the class count and the label list differ.
The app can download a model from an address at run time (Manage models, 5.7): the "raw YOLOX" `.tflite` files can be added as they are, and the `.onnx` ones run on the CPU without conversion
(convert them with 6.1 to use the GPU). Only 1-class and 3-class files below need a class names address; the labels file for the Autoware detector is listed with it.

**Drop-in, raw YOLOX output (Apache-2.0)**

| Model | Classes / input | Download |
|---|---|---|
| YOLOX-nano, COCO, ready-made GPU-clean TFLite (FP16, 2.1 MB) | 80 / 416 | https://huggingface.co/litert-community/yolox-nano-litert/resolve/main/yolox_nano.tflite |
| YOLOX-Tiny / S / M, the same kind of ready-made TFLite (9.9 / about 36 / 48.7 MB) | 80 / 416, 640, 640 | https://huggingface.co/litert-community/yolox-tiny-litert, `.../yolox-s-litert`, `.../yolox-m-litert` (file `yolox_<size>.tflite`) |
| Official YOLOX ONNX nano (3.5 MB), M (96.6 MB), L (206.7 MB), Darknet (243.0 MB) (convert with 6.1) | 80 | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx (also `yolox_m.onnx`, `yolox_l.onnx`, `yolox_darknet.onnx`) |
| **US licence plates**, YOLOX-S fine-tuned on about 20k images, AP50:95 90.1 (34.1 MB) | 1 (`license_plate`) / 640 | https://huggingface.co/autolane/yolox-s-alpr/resolve/main/yolox_plates_s.onnx (card: https://huggingface.co/autolane/yolox-s-alpr; the int8 TFLite there is not GPU-friendly, convert the ONNX) |
| **Traffic lights**, Autoware whole-image detector, YOLOX-S (34.1 MB) | 3 (`BACKGROUND`, `traffic_light`, `pedestrian_traffic_light`) / 960 | https://huggingface.co/AutowareFoundation/tensorrt_yolox/resolve/main/yolox_s_car_ped_tl_detector_960_960_batch_1.onnx (labels: `.../car_ped_tl_detector_labels.txt`) |
| **Traffic light fine detector**, TIER IV YOLOX-S (34.1 MB), meant for refining ROIs but runs on whole images | 3 / 416 | https://huggingface.co/AutowareFoundation/traffic_light_fine_detector/resolve/main/tlr_car_ped_yolox_s_batch_1.onnx |
| **Body / head / hand**, PINTO YOLOX family (Apache-2.0 repo; the TFLite-only nano archive is 9.7 MB; other archives are 0.2-14 GB and mix many formats; not inspected, check the layout and quantisation) | 3 (body, head, hand) | https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/426_YOLOX-Body-Head-Hand/resources_n_tflite.tar.gz (folder: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/426_YOLOX-Body-Head-Hand) |

Autoware also publishes generic vehicle/pedestrian detectors (`yolox-sPlus-T4-960x960-pseudo-finetune.onnx`, 56.6 MB, classes CAR, TRUCK, BUS, BICYCLE, MOTORBIKE, PEDESTRIAN, ANIMAL; not
inspected, the s-Plus variants may differ) in the same repository; its `yolox-tiny.onnx` has a TensorRT NMS plugin attached and will not run here.

**Other architectures (need a new decoder; output layouts not verified here)**

| Model | Notes | Download |
|---|---|---|
| MediaPipe EfficientDet-Lite0 (13.2 MB; int8 4.4 MB) and Lite2 (22.0 MB), COCO | TFLite, Apache-2.0 according to a third-party wrapper, confirm on Google's model card | https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite and `.../efficientdet_lite2/float32/1/efficientdet_lite2.tflite` |
| MediaPipe SSD MobileNet V2 (10.8 MB), COCO | TFLite | https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/1/ssd_mobilenet_v2.tflite |
| BlazeFace short range, faces (0.2 MB) | TFLite, MediaPipe | https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite |
| YuNet, faces (0.2 MB) | ONNX from OpenCV Zoo (MIT there); a TFLite mirror says BSD-3 | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx, https://huggingface.co/litert-community/YuNet-Face-LiteRT |
| RT-DETRv2, D-FINE, RF-DETR (no NMS step), COCO | Apache-2.0; LiteRT versions are split in two graphs | https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX, https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT, https://huggingface.co/litert-community/D-FINE-S-LiteRT, https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT |
| RT-DETR fine-tuned on licence plates | Apache-2.0 | https://huggingface.co/Topurrra/rtdetr-license-plate-detection-onnx |
| SSDLite MobileNetV3 320 | BSD-3 | https://huggingface.co/mlboydaisuke/ssdlite320-mobilenetv3-litert |
| Snow-throw sign detector, YOLOX-Tiny fine-tuned (Japanese motorway signs), outputs `boxes` and `scores` for 200 candidates, input 1280x736 | Apache-2.0; not raw YOLOX | https://huggingface.co/infra-intelligence/snow-throw-sign-yolox-tiny |

**Licence traps.** Hugging Face has many fine-tuned detectors (faces, PPE, helmets, wildlife, fire/smoke) whose cards say Apache-2.0, MIT or CC BY 4.0 but whose names or model
cards show they were trained with Ultralytics YOLOv8 / YOLO11 / YOLO26 (for example `AdamCodd/YOLOv11n-face-detection`, `UWyo/wildlife-*`, `Hexmon/vyra-yolo-ppe-detection`,
`pyronear/yolo11s_sensitive-detector`). Ultralytics code and weights are AGPL-3.0, and a card's own licence tag does not change what the model was derived from, so avoid them for an app
that may become closed or paid. Datasets have their own terms too (for example VisDrone for the UAV YOLOX-Nano-P2 checkpoints, MIT but `.pth` only and not raw YOLOX).

**To get a model for another kind of detection:** fine-tune YOLOX yourself from the official COCO weights (https://yolox.readthedocs.io/en/latest/train_custom_data.html), export to ONNX
and convert as in 6.1; that keeps Apache-2.0 end to end.

### 6.4 Tried and discarded

- **YOLO11 (AGPL-3.0)** was the original model (`yolo11s`, `yolo11s_480`) and is gone from this repository for licensing reasons; its history stays in the old repository
  (https://github.com/jgomezsa/android_Yolo11_onnx), including these experiments:
  - **ONNX Runtime with NNAPI/XNNPACK** as the fallback: ORT for Android has no GPU backend; NNAPI sent the model to the MediaTek APU and compiled it for 6 s (fp32/fp16) or 19-29 s
    (INT8) on every start because ORT cannot cache; XNNPACK on the CPU took 220-400 ms per frame and about 3 cores. Replaced by LiteRT's own CPU accelerator, which needs no second
    model file and no ONNX Runtime dependency.
  - **INT8 quantisation** (QDQ, Conv-only, 32 calibration images): kept 95% of the detections but NNAPI's compile time made it unusable.
  - **MediaTek NeuroPilot via LiteRT**: the supported SoC list does not include the Dimensity 8400 and it needs ahead-of-time compilation and Play packaging.
  - **Automatic switch to a lighter model** when below 3 FPS: removed now that the user picks the model.
- **CameraX** (previous camera layer): replaced by Camera2 to reach the ultrawide lens; see 5.5.

## 7. Measurements (Poco X7 Pro)

Milliseconds from the `Inference` log. "run" is the model, "frame" is the whole image-thread pass including the camera conversion (YUV to rotated bitmap, about 15 ms at 1280x960).

| Backend | Model | Load | run ms | frame ms | notes |
|---|---|---|---|---|---|
| GPU | YOLOX-S 640 | 1.1-1.2 s cold, 0.17-0.24 s cached | 45-60 | 63-125 | about 10-15 fps |
| GPU | YOLOX-Tiny 416 | 1.4 s cold, 0.24 s cached | 20-30 | 36-50 | about 20-25 fps |
| CPU (LiteRT, XNNPACK) | YOLOX-Tiny 416 | 0.16 s | about 76 | about 95 | about 10 fps |
| CPU (LiteRT, XNNPACK) | YOLOX-S 640 | 0.4 s | about 290 | about 310 | about 3 fps |
| CPU (ONNX Runtime) | YOLOX-S-sized 640 (downloaded `yolox_plates_s.onnx`) | 0.7 s | about 286 | about 409 | about 2.5 fps |
| GPU | `yolox_nano.tflite` 416 (downloaded) | 0.3 s | not split out | 21-25 shown | the time shown on screen (preprocess + model + decode) |

- Numbers vary with scene, temperature and background load; the phone was often charging and warm. Decode takes under 3 ms, preprocessing 3-10 ms.
- Camera conversion and the Camera2 path cost a bit more than CameraX did, but the 0.6x lens is reachable.
- Published COCO accuracy (AP, from memory): YOLOX-S 40.5, YOLOX-Tiny about 32.8; YOLO11s was 47.0 (the price of the permissive licence).
- There is no load-time limit any more (the user picks the model); the GPU path still caches compiled programs so only the first start of each model pays the compilation.

## 8. Icons

### 8.1 Launcher icon

Adaptive icon (API 26+, no PNGs): `ic_launcher_background.xml` (dark gradient), `ic_launcher_foreground.xml` (two detection boxes, teal and amber, each with a pointed
label tag containing a text bar; a person inside the teal box and a ball inside the amber one) and `ic_launcher_monochrome.xml` (themed icons, Android 13+).
`mipmap-anydpi-v26/ic_launcher.xml` and `ic_launcher_round.xml` and the manifest entries are hand-written.

The three drawables are generated from one geometry description by `tools/make_icon.py`, which also renders a preview with circle / rounded-square masks and reports how far the
artwork reaches from the centre (it must stay within 33 dp: the adaptive-icon safe zone is a 66 dp circle):

```powershell
uv run --python 3.12 --with pillow python tools/make_icon.py app/src/main/res preview.png
```

Design notes: a plain tab above a box reads as a folder, so the tags are pointed (price-tag shape); objects inside the boxes make it read as "detections".

### 8.2 Other drawables

`res/drawable/ic_chip.xml` (GPU badge), `ic_play.xml`, `ic_stop.xml`, `ic_record.xml` are hand-drawn 24 dp vectors, white, tinted in Compose.

## 9. Testing

`.\gradlew.bat test` runs the JVM unit tests: `YoloPostProcessorTest` (letterbox, box mapping, NMS, IoU), `YoloxPostProcessorTest` (grid/stride decode for 640 and 416, score, NMS),
`ModelsTest` (built-in models, format detection from magic bytes, labels, https-only addresses), `ModelContractTest` (shape and value checks, YOLOv8-style and decoded outputs rejected),
`ModelStoreTest` (install, list, unique ids and names, delete safety, damaged folders), `ModelDownloaderTest` (progress, redirects, HTTP errors, size limit, cancel; a tiny `ServerSocket` HTTP server,
because the Kotlin test compiler cannot see the JDK's `HttpServer`),
`BackendChainTest` (GPU first, CPU fallback, all-fail), `CameraPickerTest` (camera order), `YuvConverterTest` (colours, chroma layouts, rotation), `ZoomPresetsTest` (stops per range,
selector mapping, labels, model names), `FrameTimeHistoryTest` (window, `niceMax`), `RecordingFormatTest` (even size, bit rate, file name, elapsed time).
Detection quality, speed, the camera and the UI are checked on the phone with `adb logcat` and screenshots (taps with `adb shell input tap`). Add a unit test for any new logic.

## 10. Licensing

- **The app's own code** is under the PolyForm Noncommercial License 1.0.0 (file `LICENSE`, Required Notice: Copyright (c) 2026 Jorge Raúl Gómez): anyone may read, run and modify it for noncommercial
  purposes, and any commercial use needs the author's permission. The author can keep publishing builds (free now, possibly paid later) and can later make the repository private; copies already
  published stay available under this licence. Outside contributions need a written agreement if they are to be used in a closed version. Not an OSI open-source licence (it is source-available).
- **YOLOX** (code and the released weights): Apache-2.0. Shipping it needs the licence text and attribution, and a note that the model was converted (modified).
- **LiteRT, Jetpack Compose and the other AndroidX libraries:** Apache-2.0. **COCO annotations / class names:** CC BY 4.0 (attribution).
- **Not used any more:** Ultralytics YOLO11 (AGPL-3.0); the new repository must never contain YOLO11 files in its history. Avoid YOLOv5/8/10/11 (AGPL), YOLOv7/v9 (GPL-3.0),
  YOLO-NAS (non-commercial weights); YOLOv4's pretrained weights are described as demo-only, and it is heavier and less accurate.
- Other permissively licensed models to evaluate later (verify the exact weights licence first): RT-DETR (the original Apache-2.0 repository, not Ultralytics' copy), PP-YOLOE,
  D-FINE, EfficientDet-Lite and SSD-MobileNet (Apache-2.0), RF-DETR (Apache-2.0 for most sizes).
- Not legal advice; check the current terms before publishing.

## 11. Known issues and gotchas

- **Gradle `Couldn't delete ...\R.jar` (Windows).** The VS Code Kotlin language server and stray Gradle daemons lock build files. Kill all `java.exe` processes and delete
  `app\build\intermediates\compile_and_runtime_not_namespaced_r_class_jar`, then rebuild; the language server respawns, so do both in one command.
- **IDE diagnostics are unreliable.** The language server does not know about `-Xskip-metadata-version-check` ("incompatible version of Kotlin" on LiteRT classes) and is often stale right
  after an edit (including "Unresolved reference" on `R`). Gradle is the truth.
- **PowerShell 5.1 and non-ASCII source.** `Get-Content`/`Set-Content` without an explicit UTF-8 encoding corrupt characters such as "…", "▾" and "✓" (mojibake) and may add a BOM;
  edit Kotlin files with the editor tools or .NET `ReadAllText`/`WriteAllText` with UTF-8.
- **Stale Kotlin incremental compilation.** After large refactors the build can fail with "Daemon compilation failed: null" or with "Unresolved reference" to things that exist. Fix: stop Gradle and Java
  processes, delete `app\build\kotlin` and `app\build\tmp\kotlin-classes`, and rebuild with `.\gradlew.bat installDebug "-Pkotlin.compiler.execution.strategy=in-process"` (quote the property in PowerShell).
- **Platform `Dialog` and the keyboard.** A Compose `Dialog` window did not get the keyboard inset on the Poco X7 Pro, hiding its bottom buttons; use an overlay in the main window with
  `WindowInsets.safeDrawing` (5.7). `adb shell input text` works for typing addresses into text fields; clear a field with many `keyevent 67`.
- **Debug APK size.** About 342 MB: LiteRT and ONNX Runtime native libraries for four CPU architectures plus 56 MB of models. A release build or bundle splits by architecture; an `abiFilters`
  setting for `arm64-v8a` would shrink the APK for phones only.
- **LiteRT 2.3 and Kotlin.** It is built with a newer Kotlin than 2.0.21; compiling works only with the skip-metadata flag in `app/build.gradle.kts`. Upgrading the Kotlin toolchain
  would remove the flag.
- **Dynamic-shape TFLite models** make the GPU path fail (the app then uses the CPU); the reason is in `adb logcat -s Inference:W`.
- **Hidden cameras are vendor-specific.** Opening an id that is not in `cameraIdList` works on the Poco X7 Pro but may be refused or behave differently on other phones; the picker
  falls back to the next camera if opening or configuring fails.
- **Edge taps.** Zoom points too close to the screen edge can be swallowed by the system back gesture; the selector keeps 60 dp from the edge.
- **First start.** The cold GPU compile takes a few seconds ("Loading…" is shown); a cleared cache brings it back.
- Release signing is not configured; only debug builds exist.
