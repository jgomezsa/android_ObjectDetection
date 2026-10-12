# Android Object Detection (YOLOX)

Point your phone's camera at the world and the app draws a box and a label around every object it recognises,
live, using an AI model that runs entirely **on the phone**. No server is needed and no image ever leaves the device;
the app only uses the internet when you ask it to download a model.

- Recognises the 80 everyday object classes of the COCO dataset (person, car, bottle, laptop, cat, ...)
- Two models to choose from: **YOLOX-S** (more accurate) and **YOLOX-Tiny** (faster)
- Runs on the phone's **GPU** when it can, and on the CPU when it cannot
- **Add your own models** (`.tflite` or `.onnx`) from a web address: YOLOX, YOLOv8, YOLO11, YOLOv10 and DETR-style detectors such as RT-DETR, for example for licence plates or traffic lights; delete them again at any time
- Zoom from the ultrawide lens up to 10x on phones that allow it
- Records videos with the boxes drawn on
- Kotlin, Jetpack Compose, Camera2, LiteRT and ONNX Runtime; tested on a Poco X7 Pro (Android 15)

## What you need

| Requirement | Notes |
|---|---|
| Android phone | Android 8.0+ (API 26), with a camera and a USB cable |
| JDK 17 | Set `JAVA_HOME`. Tested with Microsoft OpenJDK 17 |
| Android SDK | `platform-tools`, `platforms;android-35`, `build-tools;35.0.0` (command-line tools are enough; Android Studio is optional) |
| Git | To clone the repo |

Gradle 8.9 is downloaded automatically by the wrapper (`gradlew`). The project was developed on Windows 11
with PowerShell; on macOS/Linux use `./gradlew` instead of `.\gradlew.bat`.

### Install the Android SDK (command line only)

```powershell
# After installing the Android command-line tools and putting cmdline-tools\latest\bin on PATH:
sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0"
sdkmanager --licenses
```

Create `local.properties` in the project root pointing to your SDK (it is git-ignored). Android Studio does this for you:

```properties
sdk.dir=C\:\\Users\\<you>\\AppData\\Local\\Android\\Sdk
```

## Prepare the phone

1. **Settings > About phone**, tap **Build number** 7 times to enable Developer options.
2. **Developer options**: enable **USB debugging**.
   - On **Xiaomi / Poco / Redmi (HyperOS, MIUI)** also enable **Install via USB** and
     **USB debugging (Security settings)**, otherwise installs fail with `INSTALL_FAILED_USER_RESTRICTED`.
     Keep the phone unlocked and accept the install prompt when it appears.
3. Connect the phone over USB and accept the RSA fingerprint prompt.
4. Check it shows up as `device`:

```powershell
adb devices
```

## Build and run

```powershell
git clone https://github.com/jgomezsa/android_ObjectDetection.git
cd android_ObjectDetection

.\gradlew.bat test                 # unit tests (host JVM, no phone needed)
.\gradlew.bat assembleDebug        # -> app\build\outputs\apk\debug\app-debug.apk
.\gradlew.bat installDebug         # build + install on the connected phone
adb shell am start -n com.example.objectdetection/.MainActivity
```

You can also open the folder in Android Studio and press Run.

## Using the app

1. Open **Object Detection** and allow the **Camera** permission.
2. Point the camera at objects. Each one gets a coloured box with its name and a confidence percentage.
3. The number in the top-left corner is how long the last frame took to process, in milliseconds. The lower it is, the smoother the detection. The small graph to its left shows the same time for the last 50 frames.
4. The top-right corner shows the model in use. **Tap its name** to open the model menu and switch to another model; your choice is remembered. Loading a model can take a few seconds
   the first time ("Loading..." is shown). The last entry, **Manage models…**, lets you add and delete models (see [Adding your own models](#adding-your-own-models)). Below the name, the chip icon tells you whether the phone's GPU is doing the work (teal **GPU**) or the app had to use the CPU (amber **No GPU**, slower).
5. Below the picture, the **Confidence** slider (1% to 99% in steps of 1%, remembered) sets how sure the model must be before a box is shown: lower it to see more boxes, raise it to see only the sure ones. Under it, the slider with points (for example 0.6x, 1x, 2x, 5x, 10x) sets the camera zoom: tap a point, slide along the line, or pinch on the picture. The points
   depend on what your phone's camera can do; 0.6x is the ultrawide lens on phones that make it available to apps.
6. The **Stop** button below the picture pauses the detection (the camera keeps showing the live view, and no boxes are drawn). Press **Start** to resume.
7. The **Record** button next to it saves a video of what the camera sees, with the boxes drawn on. While recording it shows the elapsed time; press it again to stop. The video is saved in the **Movies/ObjectDetection** folder of your phone (it also shows up in your gallery). Pausing the detection also ends the recording.

There is nothing else to configure: for each model the app tries the GPU first and uses the CPU if the model does not run there.

## Adding your own models

Open the model menu and choose **Manage models…**. The list shows every model: the built-in **YOLOX-S** and **YOLOX-Tiny** (they cannot be deleted) and the ones you added (each has a **Delete**
button). Under **Add a model from a web address**, type the address of a model file (it must start with `https://`), optionally a name for the model (otherwise the file name is used) and the address of a text file with the class names, and press
**Download and add**. The app downloads the file, finds out its format, loads it once to check that it fits, and only then adds it to the list; if something is wrong, it tells you what.
A model you add is not selected automatically: pick it from the model menu. Models are stored in the app's private storage and are removed when you delete them (or uninstall the app).

**Model formats and their caveats**

- **`.tflite`** (TensorFlow Lite / LiteRT) runs on the **GPU** (fast) or the CPU. It must have a fixed input size and use float numbers; `int8` (quantised) files usually cannot use the GPU and
  may not load at all. Models published as ONNX can be converted to `.tflite` on a PC (see the technical documentation); the phone cannot convert them.
- **`.onnx`** runs on the **CPU only** (ONNX Runtime), so it is several times slower than a `.tflite` on the GPU (a YOLOX-S sized model reaches about 2 to 3 frames per second), but it needs no conversion.
- **Supported model types.** The app finds out the type by running the model once and looking at what it returns; each type gets its own colour order, value range and resizing:
  **YOLOX** (raw `[1, anchors, 5 + classes]`), **YOLOv8 / YOLO11** (`[1, 4 + classes, 8400]`, boxes in pixels or as fractions), **NMS-free YOLO** such as YOLOv10 (`[1, 300, 6]`) and **DETR-style**
  detectors such as RT-DETR and D-FINE (class logits plus boxes, or `[1, 300, 4 + classes]`). The model takes a single square picture.
- **Not supported:** models with extra inputs, inputs normalised the ImageNet way (the original DETR, RF-DETR), non-square inputs, SSD, EfficientDet, segmentation, pose or classification models, and archives
  (`.zip`, `.tar.gz`: download the model file itself). Most YOLOv8/v10/v11 models are licensed under the AGPL, which matters if you plan to sell an app.
- **Class names:** a model with 80 classes uses the COCO names. For any other number of classes, give the address of a `.txt` file with one name per line (the number of lines must match the
  number of classes); without it the boxes are labelled "class 0", "class 1"...
- **Size and trust:** files up to 400 MB are accepted. Only download models you trust, and check their licence.

Examples of addresses to try (Apache-2.0 models; the technical documentation lists more):

```
https://huggingface.co/litert-community/yolox-nano-litert/resolve/main/yolox_nano.tflite
https://huggingface.co/autolane/yolox-s-alpr/resolve/main/yolox_plates_s.onnx
```

## How it works

Every camera frame goes through the same steps:

```
Camera frame -> resize and pad to a square -> AI model on the GPU (or CPU) -> raw predictions
             -> decode, keep the confident ones, remove duplicates -> boxes drawn over the same picture
```

1. **Camera.** The app reads the camera with Android's Camera2 interface. It looks for the back camera that can zoom out the furthest (some phones hide their
   ultrawide camera from simple camera libraries) and falls back to the normal one if that cannot be opened. The picture on screen is the very frame that was analysed, so the
   boxes always match what you see, even on moving objects. If the analysis falls behind, camera frames are simply skipped.
2. **Preparation.** The camera's YUV image is converted to colour and rotated upright. The model expects a square picture (640x640 for YOLOX-S, 416x416 for YOLOX-Tiny): the frame is
   scaled to fit inside it and the unused space is filled with grey so nothing is stretched.
3. **The model.** [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is a neural network trained on the COCO dataset. In one pass it looks at the whole picture and proposes
   thousands of candidate boxes, each with a score for every one of the 80 classes. It runs through
   [LiteRT](https://ai.google.dev/edge/litert) (Google's on-device runtime, the successor of TensorFlow Lite) on the phone's GPU, or on its CPU.
4. **Cleaning up the result** (`YoloxPostProcessor`). The raw numbers are turned into boxes, candidates scoring below 25% are dropped, and when several boxes cover the same object
   the best one is kept and the others that overlap it by more than 45% are discarded (*non-maximum suppression*, applied per class). The boxes are then mapped back from the padded square to the real camera picture.
5. **Drawing.** The boxes and labels are painted on a Compose canvas on top of the analysed camera picture.

Some details that keep it fast and robust:

- **GPU first, CPU as the fallback.** For every `.tflite` model the app tries the GPU. If the model cannot be compiled or run there (it even does a test run to make sure), it uses the CPU instead.
  YOLOX-Tiny is fast enough on the CPU to be usable; YOLOX-S is slow there. `.onnx` models always run on the CPU.
- **GPU program cache.** Before the first run the GPU needs the model translated into programs for that specific chip, which takes a second or two.
  The result is saved in the app's cache folder, so later starts skip that step. If the model file changes, the cache key changes with it.
- **Everything is local.** The models are bundled inside the app; nothing is downloaded or uploaded.

### Project layout

| File | Role |
|---|---|
| `MainActivity.kt` | Edge-to-edge Compose host and the camera permission request |
| `CameraScreen.kt` | The screen: picture, model selector, zoom slider, buttons; loads models with the GPU-then-CPU fallback |
| `Camera2Source.kt` | Opens the camera (Camera2), delivers upright frames, handles zoom |
| `CameraPicker.kt` | Decides which camera to try first (unit tested) |
| `YuvConverter.kt` | Converts the camera image to colour and rotates it (unit tested) |
| `BackendChain.kt` | GPU first, then CPU: tries each until one works (unit tested) |
| `LiteRtDetector.kt` | Runs a `.tflite` model with LiteRT on the GPU or the CPU, with the GPU program cache |
| `OrtDetector.kt` | Runs an `.onnx` model with ONNX Runtime on the CPU |
| `ObjectDetector.kt` | Common interface for detectors, and the code that pads a frame into the square model input |
| `YoloxPostProcessor.kt` | Turns YOLOX's raw output into boxes (unit tested) |
| `YoloPostProcessor.kt` | Shared helpers: coordinate mapping, overlap removal (unit tested) |
| `Models.kt` | Model formats and detection of the format from a file's first bytes, the model description, the built-in models (unit tested) |
| `ModelContract.kt` | Checks that a model fits the YOLOX contract and explains why not (unit tested) |
| `ModelStore.kt` | Stores downloaded models, class names, web address checks (unit tested) |
| `ModelDownloader.kt` | Downloads a model over https with a size limit and cancel (unit tested against a local server) |
| `ModelInstaller.kt` | Download, check and install a model |
| `ManageModelsDialog.kt` | The "Manage models" screen: list, delete, add from an address, format notes |
| `ZoomPresets.kt` | Works out which zoom points the camera supports (unit tested) |
| `VideoRecorder.kt` | Saves the video with the boxes drawn on |
| `DetectionRenderer.kt` | Draws the boxes and labels (on screen and in the video) |
| `RecordingFormat.kt` | Small helpers for the recording (unit tested) |
| `FrameTimeHistory.kt` | Keeps the last 50 frame times for the graph (unit tested) |
| `CocoLabels.kt` | The 80 class names |
| `app/src/main/assets/` | The bundled models: `yolox_s.tflite` and `yolox_tiny.tflite` |

More technical detail (architecture, the camera, models and how to convert them, measurements, licensing): [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| `INSTALL_FAILED_USER_RESTRICTED` | Enable **Install via USB** in Developer options (Xiaomi/Poco/Redmi) and accept the prompt on the phone |
| `adb devices` shows `unauthorized` or nothing | Re-plug the cable, accept the RSA prompt, set USB mode to File transfer |
| `SDK location not found` | Create `local.properties` with `sdk.dir=...` |
| Gradle `Couldn't delete ...\R.jar` (Windows) | Something locks build files, usually an IDE language server or a stray Gradle daemon. Run `.\gradlew.bat --stop`, close the IDE, delete `app\build`, rebuild |
| The app crashes on start | `adb logcat -s AndroidRuntime:E` |
| The picture is black | Make sure the camera permission is granted and no other app is using the camera |

## Third-party components and licences

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii) models, Apache-2.0. The bundled files are the official YOLOX-S and YOLOX-Tiny ONNX models converted to TFLite
  (`onnx2tf`, static shapes), so they are modified copies.
- **LiteRT, Jetpack Compose and the AndroidX libraries**, Apache-2.0. **ONNX Runtime** (Microsoft), MIT.
- **Models you download** have their own licences (some YOLOv8/v11 models are AGPL-3.0); you are responsible for checking them.
- **COCO** class names and the dataset the models were trained on: annotations are CC BY 4.0 ([cocodataset.org](https://cocodataset.org/#termsofuse)).

The app's own code is under the [PolyForm Noncommercial License 1.0.0](LICENSE): you may read, run and modify it for noncommercial purposes (personal use, study, research, hobby projects,
nonprofits); any commercial use needs the author's permission. The third-party components above keep their own licences. Check the current terms of each component before
publishing a build; this is not legal advice.
