package com.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

/**
 * Runs a TFLite model (NHWC float input `[1, size, size, 3]`) with LiteRT on the given [backend]: the GPU (compiled program cached in the
 * app's cache directory, so only the first start of a model pays the compilation) or the CPU (XNNPACK). The model is a bundled asset
 * (`<id>.tflite`) or a downloaded file, and its output can be any supported kind (see [OutputLayout]). Creation runs the model once: for a
 * model that is being checked for the first time ([ModelInfo.spec] is null) this recognises the kind of model ([ModelSpec]); otherwise it
 * verifies that the model still behaves as recorded. It throws if the backend cannot run the model, so the caller can move on to the next one.
 *
 * The model must have static tensor shapes: the GPU delegate rejects dynamic ones (see TECHNICAL_DOCUMENTATION.md for how to convert).
 */
class LiteRtDetector(context: Context, private val info: ModelInfo, override val backend: Backend) : ObjectDetector {
    private val env = Environment.create(context)

    // lateinit so that close() can release whatever was created when creation fails halfway.
    private lateinit var model: CompiledModel
    private lateinit var inputBuffers: List<TensorBuffer>
    private lateinit var outputBuffers: List<TensorBuffer>
    private lateinit var renderer: LetterboxRenderer
    private lateinit var input: FloatArray
    private lateinit var names: List<String>

    /** What kind of model this is; for a model that was just added it is worked out here. */
    lateinit var spec: ModelSpec
        private set

    init {
        try {
            val options = when (backend) {
                Backend.GPU -> {
                    // The cache key includes the model size, so replacing a model file never picks up a stale compiled program.
                    val gpu = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP16,
                        serializationDir = context.cacheDir.absolutePath,
                        modelCacheKey = "${info.id}-${info.sizeBytes}",
                        serializeProgramCache = true,
                    )
                    CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = gpu }
                }
                Backend.CPU -> CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6))
                }
            }
            val file = info.file
            model = if (file == null) {
                CompiledModel.create(context.assets, "${info.id}.tflite", options, env)
            } else {
                CompiledModel.create(file.absolutePath, options, env)
            }
            inputBuffers = model.createInputBuffers()
            outputBuffers = model.createOutputBuffers()
            val inputSize = ModelContract.tfliteInputSize(inputBuffers[0].readFloat().size)
            input = FloatArray(inputSize * inputSize * 3)
            val known = info.spec
            spec = if (known == null) {
                ModelContract.resolve(inputSize, outputBuffers.map { null }) { value -> run(value) }
            } else {
                check(known.inputSize == inputSize) { "The model input is ${inputSize}x$inputSize, expected ${known.inputSize}" }
                ModelContract.verify(known, run(ModelContract.grayFor(known)))
                known
            }
            renderer = LetterboxRenderer(inputSize)
            names = LabelParser.namesFor(spec, info.labels)
        } catch (e: Throwable) {
            runCatching { close() }
            throw e
        }
        Log.d(TAG, "${info.id} ready on the ${backend.label} (${spec.family}, input ${spec.inputSize}x${spec.inputSize}, ${spec.numClasses} classes)")
    }

    /** Runs the model on a constant image of [value] and returns the output tensors. */
    private fun run(value: Float): List<FloatArray> {
        input.fill(value)
        inputBuffers[0].writeFloat(input)
        model.run(inputBuffers, outputBuffers)
        return outputBuffers.map { it.readFloat() }
    }

    override fun detect(bitmap: Bitmap, threshold: Float): DetectionResult {
        val t0 = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, spec.inputSize, spec.stretch)
        InputPacker.nhwc(renderer.render(bitmap, letterbox), spec.convention, input)
        inputBuffers[0].writeFloat(input)
        val t1 = System.nanoTime()

        model.run(inputBuffers, outputBuffers)
        val raw = outputBuffers.map { it.readFloat() }
        val t2 = System.nanoTime()

        val detections = OutputDecoders.decode(spec, raw, letterbox, names, threshold)
        val t3 = System.nanoTime()
        return DetectionResult(
            detections,
            inferenceMs = (t3 - t0) / 1_000_000,
            preprocessMs = (t1 - t0) / 1_000_000,
            runMs = (t2 - t1) / 1_000_000,
            postMs = (t3 - t2) / 1_000_000,
        )
    }

    override fun close() {
        if (::inputBuffers.isInitialized) inputBuffers.forEach { it.close() }
        if (::outputBuffers.isInitialized) outputBuffers.forEach { it.close() }
        if (::model.isInitialized) model.close()
        env.close()
    }

    private companion object {
        const val TAG = "LiteRtDetector"
    }
}
