package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Runs an ONNX model (NCHW float input `[1, 3, size, size]`) with ONNX Runtime on the CPU; the output can be any supported kind
 * (see [OutputLayout]). ONNX Runtime for Android has no GPU backend, so these models are slower than TFLite ones on the GPU, but a downloaded
 * `.onnx` file needs no conversion. Creation runs the model once: for a model that is being checked for the first time
 * ([ModelInfo.spec] is null) this recognises the kind of model ([ModelSpec]); otherwise it verifies that the model still behaves as recorded.
 */
class OrtDetector(@Suppress("UNUSED_PARAMETER") context: Context, private val info: ModelInfo, override val backend: Backend = Backend.CPU) : ObjectDetector {
    private val env = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    private lateinit var inputName: String
    private lateinit var renderer: LetterboxRenderer
    private lateinit var inputBuffer: FloatBuffer
    private lateinit var names: List<String>
    private var size = 0

    /** What kind of model this is; for a model that was just added it is worked out here. */
    lateinit var spec: ModelSpec
        private set

    init {
        require(backend == Backend.CPU) { "ONNX models run on the CPU only" }
        try {
            val file = info.file ?: throw ModelRejectedException("ONNX models have to be downloaded files.")
            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 6))
            }
            session = env.createSession(file.absolutePath, options)
            if (session.inputNames.size != 1) {
                throw ModelRejectedException("The model has ${session.inputNames.size} inputs; only models with a single image input are supported.")
            }
            inputName = session.inputNames.first()
            size = ModelContract.onnxInputSize((session.inputInfo[inputName]!!.info as TensorInfo).shape)
            inputBuffer = ByteBuffer.allocateDirect(4 * 3 * size * size).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val known = info.spec
            spec = if (known == null) {
                val shapes = session.outputNames.map { name -> (session.outputInfo[name]?.info as? TensorInfo)?.shape }
                ModelContract.resolve(size, shapes) { value -> run(value) }
            } else {
                check(known.inputSize == size) { "The model input is ${size}x$size, expected ${known.inputSize}" }
                ModelContract.verify(known, run(ModelContract.grayFor(known)))
                known
            }
            renderer = LetterboxRenderer(size)
            names = LabelParser.namesFor(spec, info.labels)
        } catch (e: Throwable) {
            runCatching { close() }
            throw e
        }
        Log.d(TAG, "${info.id} ready on the CPU (ONNX Runtime, ${spec.family}, input ${size}x$size, ${spec.numClasses} classes)")
    }

    /** Runs the model on a constant image of [value] and returns the output tensors. */
    private fun run(value: Float): List<FloatArray> {
        for (i in 0 until inputBuffer.capacity()) inputBuffer.put(i, value)
        return runOnInput()
    }

    /** Runs the model on the current [inputBuffer] and returns every output flattened. */
    private fun runOnInput(): List<FloatArray> {
        inputBuffer.rewind()
        return OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, size.toLong(), size.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                (0 until result.size()).map { index ->
                    val tensorOut = result[index] as? OnnxTensor ?: throw ModelRejectedException("An output of the model is not a tensor.")
                    val out = try {
                        tensorOut.floatBuffer
                    } catch (e: Exception) {
                        throw ModelRejectedException("An output of the model is not made of 32-bit floats, which is all the app can read.")
                    }
                    out.rewind()
                    FloatArray(out.remaining()).also { out.get(it) }
                }
            }
        }
    }

    override fun detect(bitmap: Bitmap, threshold: Float): DetectionResult {
        val t0 = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, size, spec.stretch)
        InputPacker.nchw(renderer.render(bitmap, letterbox), spec.convention, inputBuffer)
        val t1 = System.nanoTime()
        val raw = runOnInput()
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
        if (::session.isInitialized) session.close()
    }

    private companion object {
        const val TAG = "OrtDetector"
    }
}
