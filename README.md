# Android Object Detection (YOLOX)

Point your phone's camera at the world and the app draws a box and a label around every object it recognises,
live, using an AI model that runs entirely **on the phone**. No internet connection and no server are needed, and
no image ever leaves the device.

- Recognises the 80 everyday object classes of the COCO dataset (person, car, bottle, laptop, cat, ...)
- Two models to choose from: **YOLOX-S** (more accurate) and **YOLOX-Tiny** (faster)
- Runs on the phone's **GPU** when it can, and on the CPU when it cannot
- Zoom from the ultrawide lens up to 10x on phones that allow it
- Records videos with the boxes drawn on
- Kotlin, Jetpack Compose and Camera2; tested on a Poco X7 Pro (Android 15)

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
4. The top-right corner shows the model in use. **Tap its name** to switch between **YOLOX-S** and **YOLOX-Tiny**; your choice is remembered. Loading a model can take a few seconds
   the first time ("Loading..." is shown). Below the name, the chip icon tells you whether the phone's GPU is doing the work (teal **GPU**) or the app had to use the CPU (amber **No GPU**, slower).
5. Below the picture, the slider with points (for example 0.6x, 1x, 2x, 5x, 10x) sets the camera zoom: tap a point, slide along the line, or pinch on the picture. The points
   depend on what your phone's camera can do; 0.6x is the ultrawide lens on phones that make it available to apps.
6. The **Stop** button below the picture pauses the detection (the camera keeps showing the live view, and no boxes are drawn). Press **Start** to resume.
7. The **Record** button next to it saves a video of what the camera sees, with the boxes drawn on. While recording it shows the elapsed time; press it again to stop. The video is saved in the **Movies/ObjectDetection** folder of your phone (it also shows up in your gallery). Pausing the detection also ends the recording.

There is nothing else to configure: for each model the app tries the GPU first and uses the CPU if the model does not run there.

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

- **GPU first, CPU as the fallback.** For every model the app tries the GPU. If the model cannot be compiled or run there (it even does a test run to make sure), it uses the CPU instead.
  YOLOX-Tiny is fast enough on the CPU to be usable; YOLOX-S is slow there.
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
| `LiteRtDetector.kt` | Runs a model with LiteRT on the GPU or the CPU, with the GPU program cache |
| `ObjectDetector.kt` | Common interface for detectors, and the code that pads a frame into the square model input |
| `YoloxPostProcessor.kt` | Turns YOLOX's raw output into boxes (unit tested) |
| `YoloPostProcessor.kt` | Shared helpers: coordinate mapping, overlap removal (unit tested) |
| `ModelName.kt` | The list of models and their display names |
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
- **LiteRT, Jetpack Compose and the AndroidX libraries**, Apache-2.0.
- **COCO** class names and the dataset the models were trained on: annotations are CC BY 4.0 ([cocodataset.org](https://cocodataset.org/#termsofuse)).

The app's own code does not have a licence file yet, so by default all rights are reserved by the author. Check the current terms of each component before publishing a
build; this is not legal advice.
