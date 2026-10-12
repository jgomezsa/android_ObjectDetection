package com.example.objectdetection

import android.content.Context
import java.io.File

/**
 * Adds a model from a web address: downloads it, finds out its format, loads it once to recognise what kind of model it is
 * (see [ModelContract]), checks the class names, and stores it. Anything that does not pass is rejected with a message for the user.
 * Call from a background thread.
 */
class ModelInstaller(
    private val context: Context,
    private val store: ModelStore,
    private val downloader: ModelDownloader = ModelDownloader(),
) {
    fun install(modelUrl: String, labelsUrl: String, progress: (message: String, fraction: Float?) -> Unit, isCancelled: () -> Boolean): ModelInfo {
        val parsed = ModelUrl.parse(modelUrl)
        val labelsParsed = labelsUrl.takeIf { it.isNotBlank() }?.let { ModelUrl.parse(it) }
        val temp = File(context.cacheDir, "download-${System.nanoTime()}.part")
        try {
            progress("Downloading…", 0f)
            downloader.download(
                parsed.url,
                temp,
                onProgress = { done, total ->
                    val mb = done / ModelDownloader.MB
                    progress(if (total > 0) "Downloading… $mb of ${total / ModelDownloader.MB} MB" else "Downloading… $mb MB", if (total > 0) done.toFloat() / total else null)
                },
                isCancelled = isCancelled,
            )

            val header = ByteArray(512)
            val read = temp.inputStream().use { it.read(header) }
            val head = header.copyOf(maxOf(read, 0))
            val format = ModelFormat.detect(head, parsed.fileName) ?: throw ModelRejectedException(ModelFormat.whyNot(head))

            val labels = labelsParsed?.let {
                progress("Downloading the class names…", null)
                LabelParser.parse(downloader.downloadText(it.url)).also { names ->
                    if (names.isEmpty()) throw ModelRejectedException("The class names file is empty.")
                }
            } ?: emptyList()

            progress("Checking the model…", null)
            val spec = probe(temp, format)
            // An NMS-free output does not say how many classes there are, so any number of names is accepted for it.
            if (labels.isNotEmpty() && spec.layout != OutputLayout.END2END && labels.size != spec.numClasses) {
                throw ModelRejectedException("The model has ${spec.numClasses} classes but the class names file lists ${labels.size}.")
            }
            return store.install(temp, format, ModelUrl.displayName(parsed.fileName), spec, labels, parsed.url)
        } finally {
            temp.delete()
        }
    }

    /** Loads the model once (on the CPU, which every model can use) and works out what kind of model it is. */
    private fun probe(file: File, format: ModelFormat): ModelSpec {
        val info = ModelInfo("probe", "probe", format, false, file.length(), emptyList(), null, file, spec = null)
        return try {
            when (format) {
                ModelFormat.TFLITE -> LiteRtDetector(context, info, Backend.CPU).use { it.spec }
                ModelFormat.ONNX -> OrtDetector(context, info, Backend.CPU).use { it.spec }
            }
        } catch (e: ModelRejectedException) {
            throw e
        } catch (e: Throwable) {
            val detail = (e.message ?: e.javaClass.simpleName).lines().first().take(200)
            throw ModelRejectedException(
                "The model could not be loaded ($detail). It may be quantised to int8, have dynamic shapes, use operations the runtime does not support, or be damaged.",
            )
        }
    }
}
