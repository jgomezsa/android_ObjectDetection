package es.jrg.objectdetection

/** Where the privacy policy is published (GitHub Pages, from the `docs` folder of the repository). */
const val PRIVACY_POLICY_URL = "https://jgomezsa.github.io/android_ObjectDetection/privacy.html"

/** The public repository with the source code. */
const val SOURCE_URL = "https://github.com/jgomezsa/android_ObjectDetection"

/** Said plainly in the app, the store listing and the README: the source can be read, but this is not an open-source licence. */
const val LICENSE_STATEMENT =
    "The source code of this app is published on GitHub so that you can read it and use it for noncommercial purposes, " +
        "under the PolyForm Noncommercial License 1.0.0. This is not an open-source licence: commercial use is not allowed without permission."

/** A third-party component that ships inside the app (or a model that does), with its licence. */
data class Notice(val name: String, val licence: String, val url: String)

object Notices {
    val all = listOf(
        Notice("YOLOX models (YOLOX-S, YOLOX-Tiny), Megvii", "Apache License 2.0", "https://github.com/Megvii-BaseDetection/YOLOX"),
        Notice("LiteRT (TensorFlow Lite), Google", "Apache License 2.0", "https://github.com/google-ai-edge/LiteRT"),
        Notice("ONNX Runtime, Microsoft", "MIT License", "https://github.com/microsoft/onnxruntime"),
        Notice("Jetpack Compose, AndroidX and Kotlin, Google and JetBrains", "Apache License 2.0", "https://developer.android.com/jetpack/androidx"),
    )
}
