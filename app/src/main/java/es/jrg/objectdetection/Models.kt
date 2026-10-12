package es.jrg.objectdetection

import java.io.File

/** Thrown with a message meant for the user when a model file or URL cannot be accepted. */
class ModelRejectedException(message: String) : Exception(message)

enum class ModelFormat(val extension: String, val label: String) {
    /** LiteRT/TensorFlow Lite: runs on the GPU or the CPU. */
    TFLITE("tflite", "TFLite"),

    /** ONNX through ONNX Runtime: runs on the CPU only. */
    ONNX("onnx", "ONNX");

    companion object {
        /** The format of a file from its first bytes (and name); null for archives, web pages and anything unknown. */
        fun detect(header: ByteArray, fileName: String): ModelFormat? {
            if (header.size >= 8 && header[4] == 'T'.code.toByte() && header[5] == 'F'.code.toByte() &&
                header[6] == 'L'.code.toByte() && header[7] == '3'.code.toByte()
            ) {
                return TFLITE
            }
            if (isArchive(header) || looksLikeHtml(header)) return null
            // An ONNX file is a protobuf ModelProto; exporters write the ir_version field first, which starts with byte 0x08.
            if (header.isNotEmpty() && (header[0] == 0x08.toByte() || fileName.lowercase().endsWith(".onnx"))) return ONNX
            return null
        }

        /** Why [detect] returned null, as a message for the user. */
        fun whyNot(header: ByteArray): String = when {
            isArchive(header) -> "That file is an archive (zip, tar or similar). Download the .tflite or .onnx file itself."
            looksLikeHtml(header) -> "That address returned a web page, not a model. Use the direct download link (on Hugging Face it contains /resolve/)."
            else -> "That file is not a TFLite or ONNX model."
        }

        private fun starts(header: ByteArray, vararg bytes: Int) =
            header.size >= bytes.size && bytes.indices.all { header[it] == bytes[it].toByte() }

        private fun isArchive(header: ByteArray) =
            starts(header, 'P'.code, 'K'.code, 3, 4) || starts(header, 0x1F, 0x8B) || starts(header, 'B'.code, 'Z'.code, 'h'.code) ||
                starts(header, '7'.code, 'z'.code, 0xBC, 0xAF) ||
                (header.size >= 262 && String(header, 257, 5, Charsets.ISO_8859_1) == "ustar")

        private fun looksLikeHtml(header: ByteArray): Boolean {
            val start = String(header, 0, minOf(header.size, 64), Charsets.ISO_8859_1).trimStart().lowercase()
            return start.startsWith("<!doctype") || start.startsWith("<html") || start.startsWith("<?xml") || start.startsWith("{")
        }
    }
}

/** A model the user can pick: bundled in the app (asset `<id>.tflite`) or downloaded into the app's storage. */
data class ModelInfo(
    val id: String,
    val displayName: String,
    val format: ModelFormat,
    val builtIn: Boolean,
    val sizeBytes: Long,
    val labels: List<String>,
    val sourceUrl: String? = null,
    /** The model file for a downloaded model; null for built-in ones. */
    val file: File? = null,
    /** What kind of model it is and how to feed and decode it; null only for a file that is being checked for the first time. */
    val spec: ModelSpec? = null,
) {
    val inputSize: Int get() = spec?.inputSize ?: 0
    val numClasses: Int get() = spec?.numClasses ?: 0
}

object BuiltInModels {
    val all = listOf(
        ModelInfo("yolox_s", "YOLOX-S", ModelFormat.TFLITE, true, 35_898_432L, CocoLabels.names, spec = ModelSpec.yolox(640, 80)),
        ModelInfo("yolox_tiny", "YOLOX-Tiny", ModelFormat.TFLITE, true, 20_260_384L, CocoLabels.names, spec = ModelSpec.yolox(416, 80)),
    )
}

/** Model loaded on the first start. */
const val DEFAULT_MODEL = "yolox_s"
