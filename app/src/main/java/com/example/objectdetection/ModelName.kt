package com.example.objectdetection

/** Models the user can choose from (asset names without extension; each needs `assets/<name>.tflite`). */
val MODELS = listOf("yolox_s", "yolox_tiny")

/** Model loaded on the first start. */
const val DEFAULT_MODEL = "yolox_s"

/** Name shown to the user for a model asset: `yolox_s` -> "YOLOX-s", `yolox_tiny` -> "YOLOX-Tiny". Other names are shown as they are. */
fun modelDisplayName(model: String): String = when (model) {
    "yolox_s" -> "YOLOX-s"
    "yolox_tiny" -> "YOLOX-Tiny"
    else -> model
}
