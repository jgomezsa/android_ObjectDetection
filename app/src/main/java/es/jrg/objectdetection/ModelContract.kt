package es.jrg.objectdetection

import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Works out what kind of model a file is by looking at its input and at its outputs on a test image, and rejects it with a reason a user can
 * understand when it is none of the supported kinds (see [OutputLayout]). Everything here is pure Kotlin so it can be unit tested.
 */
object ModelContract {
    /** Input size assumed for ONNX models whose input height and width are dynamic. */
    const val DYNAMIC_INPUT_SIZE = 640

    private const val PROB_MIN = -0.05f
    private const val PROB_MAX = 1.05f // a little slack: fp16 models give 1.003 and similar

    /** Largest raw box value expected on an undecoded YOLOX output (offsets and log sizes are small; decoded pixels are not). */
    private const val MAX_RAW_BOX = 64f
    private const val MAX_QUERIES = 2000

    /** YOLOv8-style boxes up to this are fractions of the input (Ultralytics TFLite exports); above MAX_RAW_BOX they are pixels. */
    private const val MAX_NORMALIZED = 1.5f
    private val COMMON_QUERIES = intArrayOf(300, 100, 900, 200, 500, 1000, 50)

    const val SUPPORTED = "Supported outputs: YOLOX (raw, [1, anchors, 5 + classes]), YOLOv8/YOLO11 ([1, 4 + classes, anchors]), " +
        "NMS-free YOLO ([1, 300, 6]) and DETR-style models (class logits + boxes, or [1, queries, 4 + classes])."

    /** Input side of a TFLite model from the number of values of its input tensor (a square 3-channel image whose side is a multiple of 32). */
    fun tfliteInputSize(inputElements: Int): Int {
        if (inputElements % 3 == 0) {
            val size = sqrt((inputElements / 3).toDouble()).roundToInt()
            if (size * size * 3 == inputElements && size % 32 == 0 && size >= 64) return size
        }
        throw ModelRejectedException("The model input is not a square 3-channel image ($inputElements values), so the app cannot feed it a camera picture.")
    }

    /** Input side of an ONNX model from its input shape `[batch, 3, height, width]`. Dynamic dimensions are reported as -1 (or 0). */
    fun onnxInputSize(input: LongArray): Int {
        if (input.size != 4) throw ModelRejectedException("The model input has ${input.size} dimensions; an image input has 4 ([1, 3, height, width]).")
        if (input[1] != 3L) throw ModelRejectedException("The model input is not channels-first ([1, 3, height, width]), which is what ONNX image models use.")
        val height = input[2]
        val width = input[3]
        if (height > 0 && width > 0 && height != width) throw ModelRejectedException("The model input is ${width}x$height. Only square inputs are supported.")
        val size = (if (height > 0) height else if (width > 0) width else DYNAMIC_INPUT_SIZE.toLong()).toInt()
        if (size % 32 != 0 || size < 64) throw ModelRejectedException("The model input size $size is not a multiple of 32.")
        return size
    }

    /**
     * Runs [run] on a gray image of the given value and recognises the output type. The pixel scale a model expects is unknown at this point
     * (0..255 for YOLOX, 0..1 for most others), so a gray at each scale is tried; the structure of the output does not depend on the picture.
     */
    fun resolve(inputSize: Int, shapes: List<LongArray?>, run: (Float) -> List<FloatArray>): ModelSpec {
        var firstError: ModelRejectedException? = null
        for (value in floatArrayOf(114f, 114f / 255f)) {
            try {
                return analyze(inputSize, run(value), shapes)
            } catch (e: ModelRejectedException) {
                if (firstError == null) firstError = e
            }
        }
        throw firstError!!
    }

    /** The gray value (in the model's own scale) used for self-tests of a model of this [spec]. */
    fun grayFor(spec: ModelSpec): Float = if (spec.convention == InputConvention.BGR_255) 114f else 114f / 255f

    /** Cheap check that a model whose type is already known still returns what that type needs (right sizes, no NaN). */
    fun verify(spec: ModelSpec, outputs: List<FloatArray>) {
        val expected = spec.expectedOutputSizes()
        if (outputs.map { it.size } != expected) {
            throw ModelRejectedException("The model output sizes ${outputs.map { it.size }} do not match what was found when it was added ($expected).")
        }
        if (outputs.any { out -> out.any { !it.isFinite() } }) throw ModelRejectedException("The model returned NaN or infinite values.")
    }

    /** Recognises the output type from the outputs obtained on a test image, and their ONNX shapes when known. */
    fun analyze(inputSize: Int, outputs: List<FloatArray>, shapes: List<LongArray?> = outputs.map { null }): ModelSpec {
        if (outputs.any { out -> out.any { !it.isFinite() } }) throw ModelRejectedException("The model returned NaN or infinite values.")
        return when (outputs.size) {
            1 -> analyzeSingle(inputSize, outputs[0], shapes.getOrNull(0))
            2 -> analyzeSplit(inputSize, outputs, shapes)
            else -> throw ModelRejectedException("The model has ${outputs.size} outputs; supported models have one, or two (class logits and boxes). $SUPPORTED")
        }
    }

    private fun analyzeSingle(size: Int, raw: FloatArray, shape: LongArray?): ModelSpec {
        val anchors = YoloxPostProcessor.rows(size)
        val e = raw.size
        val candidates = ArrayList<ModelSpec>()
        val dims = shape?.takeIf { it.size == 3 && it[1] > 0 && it[2] > 0 && it[1] * it[2] == e.toLong() }
        if (dims != null) {
            val a = dims[1].toInt()
            val b = dims[2].toInt()
            when {
                b == anchors && a >= 5 -> candidates += ModelSpec(OutputLayout.V8_CHANNELS_FIRST, size, a - 4)
                a == anchors -> {
                    if (b >= 6) candidates += ModelSpec(OutputLayout.YOLOX_ROWS, size, b - 5)
                    if (b >= 5) candidates += ModelSpec(OutputLayout.V8_ROWS, size, b - 4)
                }
                b == 6 && a <= MAX_QUERIES -> candidates += ModelSpec(OutputLayout.END2END, size, END2END_CLASSES, a)
                a <= MAX_QUERIES && b >= 5 -> candidates += ModelSpec(OutputLayout.DETR_MERGED, size, b - 4, a)
            }
        } else {
            if (e % anchors == 0) {
                val k = e / anchors
                if (k >= 6) candidates += ModelSpec(OutputLayout.YOLOX_ROWS, size, k - 5)
                if (k >= 5) {
                    // Channels-first is what Ultralytics exports; rows are tried after it.
                    candidates += ModelSpec(OutputLayout.V8_CHANNELS_FIRST, size, k - 4)
                    candidates += ModelSpec(OutputLayout.V8_ROWS, size, k - 4)
                }
            }
            if (e % 6 == 0 && e / 6 <= MAX_QUERIES) candidates += ModelSpec(OutputLayout.END2END, size, END2END_CLASSES, e / 6)
            for (q in COMMON_QUERIES) {
                if (e % q == 0 && e / q >= 5) candidates += ModelSpec(OutputLayout.DETR_MERGED, size, e / q - 4, q)
            }
        }
        var firstReason: String? = null
        for (candidate in candidates) {
            val reason = check(candidate, raw)
            if (reason == null) return candidate.copy(normalizedBoxes = isNormalizedYolov8(candidate, raw))
            if (firstReason == null) firstReason = reason
        }
        throw ModelRejectedException(
            (if (candidates.isEmpty()) "The model output has $e values${shape?.let { " (shape ${it.toList()})" } ?: ""}, which fits no supported type for a ${size}x$size input."
            else "The model output does not match a supported type: $firstReason") + " $SUPPORTED",
        )
    }

    /** Two outputs: class logits `[queries, classes]` and boxes `[queries, 4]` in either order. */
    private fun analyzeSplit(size: Int, outputs: List<FloatArray>, shapes: List<LongArray?>): ModelSpec {
        var reason: String? = null
        for ((li, bi) in listOf(0 to 1, 1 to 0)) {
            val logits = outputs[li]
            val boxes = outputs[bi]
            val lastBox = shapes.getOrNull(bi)?.lastOrNull()
            if (lastBox != null && lastBox > 0 && lastBox != 4L) continue
            if (boxes.size % 4 != 0) continue
            val queries = boxes.size / 4
            if (queries < 1 || queries > MAX_QUERIES || logits.size % queries != 0) continue
            val classes = logits.size / queries
            if (classes < 1) continue
            if (boxes.any { it < -0.5f || it > 1.5f }) {
                reason = "the boxes are not normalised (0 to 1)"
                continue
            }
            return ModelSpec(OutputLayout.DETR_SPLIT, size, classes, queries, li, bi)
        }
        throw ModelRejectedException("The two outputs are not class logits [1, queries, classes] plus boxes [1, queries, 4]${reason?.let { " ($it)" } ?: ""}. $SUPPORTED")
    }

    /** Null when the values fit the layout of [spec]; otherwise why not. */
    private fun check(spec: ModelSpec, raw: FloatArray): String? = when (spec.layout) {
        OutputLayout.YOLOX_ROWS -> checkRows(raw, spec.anchors, 5 + spec.numClasses, rawBoxes = true)
        OutputLayout.V8_ROWS -> checkRows(raw, spec.anchors, 4 + spec.numClasses, rawBoxes = false)
        OutputLayout.V8_CHANNELS_FIRST -> checkChannelsFirst(raw, spec.anchors, 4 + spec.numClasses)
        OutputLayout.END2END -> checkEnd2End(raw, spec.queries)
        OutputLayout.DETR_MERGED -> checkMerged(raw, spec.queries, 4 + spec.numClasses)
        OutputLayout.DETR_SPLIT -> null
    }

    /** True when a YOLOv8-style output has box values as fractions of the input, not pixels. */
    private fun isNormalizedYolov8(spec: ModelSpec, raw: FloatArray): Boolean {
        if (spec.layout != OutputLayout.V8_CHANNELS_FIRST && spec.layout != OutputLayout.V8_ROWS) return false
        val rows = spec.anchors
        val k = 4 + spec.numClasses
        var maxBox = 0f
        for (i in 0 until rows) for (c in 0 until 4) {
            val v = if (spec.layout == OutputLayout.V8_CHANNELS_FIRST) raw[c * rows + i] else raw[i * k + c]
            maxBox = maxOf(maxBox, abs(v))
        }
        return maxBox <= MAX_NORMALIZED
    }

    private fun isProbability(v: Float) = v >= PROB_MIN && v <= PROB_MAX

    private const val NOT_PROBABILITIES =
        "the objectness/class scores are not probabilities between 0 and 1 (the model must apply the sigmoid itself)"

    /** Rows of [k] values: four box values, then scores. YOLOX has small raw boxes; YOLOv8-style outputs have pixel boxes. */
    private fun checkRows(raw: FloatArray, rows: Int, k: Int, rawBoxes: Boolean): String? {
        var maxBox = 0f
        for (row in 0 until rows) {
            val base = row * k
            for (c in 0 until 4) maxBox = maxOf(maxBox, abs(raw[base + c]))
            for (c in 4 until k) if (!isProbability(raw[base + c])) return NOT_PROBABILITIES
        }
        return when {
            rawBoxes && maxBox > MAX_RAW_BOX -> "the boxes look already decoded (pixel coordinates), not raw YOLOX offsets"
            !rawBoxes && maxBox in (MAX_NORMALIZED..MAX_RAW_BOX) -> "the boxes are neither pixels nor fractions of the input"
            else -> null
        }
    }

    /** `k` channels of `rows` values each: four box channels (pixels), then one probability channel per class. */
    private fun checkChannelsFirst(raw: FloatArray, rows: Int, k: Int): String? {
        var maxBox = 0f
        for (c in 0 until 4) for (i in 0 until rows) maxBox = maxOf(maxBox, abs(raw[c * rows + i]))
        for (c in 4 until k) for (i in 0 until rows) if (!isProbability(raw[c * rows + i])) return NOT_PROBABILITIES
        return if (maxBox in MAX_NORMALIZED..MAX_RAW_BOX) "the box channels are neither pixels nor fractions of the input" else null
    }

    private fun checkEnd2End(raw: FloatArray, queries: Int): String? {
        for (row in 0 until queries) {
            val base = row * 6
            if (!isProbability(raw[base + 4])) return "the fifth column is not a score between 0 and 1"
            val cls = raw[base + 5]
            if (cls < -0.001f || cls > 100_000f || abs(cls - cls.roundToInt()) > 1e-3f) return "the sixth column is not a class number"
        }
        return null
    }

    private fun checkMerged(raw: FloatArray, queries: Int, k: Int): String? {
        for (row in 0 until queries) {
            val base = row * k
            for (c in 0 until 4) if (raw[base + c] < -0.5f || raw[base + c] > 1.5f) return "the boxes are not normalised (0 to 1)"
            for (c in 4 until k) if (!isProbability(raw[base + c])) return NOT_PROBABILITIES
        }
        return null
    }

    /** Class count assumed for NMS-free outputs, whose shape does not say how many classes there are (the labels file may have any number). */
    const val END2END_CLASSES = 80
}
