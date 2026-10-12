package es.jrg.objectdetection

import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.roundToInt

/** Turns the raw output(s) of a model of any supported [OutputLayout] into detections with normalised boxes. */
object OutputDecoders {
    const val DEFAULT_THRESHOLD = 0.25f
    private const val IOU = 0.45f

    fun decode(
        spec: ModelSpec, outputs: List<FloatArray>, lb: Letterbox, names: List<String>, threshold: Float = DEFAULT_THRESHOLD,
    ): List<Detection> = when (spec.layout) {
        OutputLayout.YOLOX_ROWS -> YoloxPostProcessor.decode(outputs[0], spec.inputSize, lb, threshold, IOU, spec.numClasses, names)
        OutputLayout.V8_CHANNELS_FIRST -> yolov8(outputs[0], spec, lb, names, threshold, channelsFirst = true)
        OutputLayout.V8_ROWS -> yolov8(outputs[0], spec, lb, names, threshold, channelsFirst = false)
        OutputLayout.DETR_SPLIT -> detrSplit(outputs[spec.logitsIndex], outputs[spec.boxesIndex], spec, lb, names, threshold)
        OutputLayout.DETR_MERGED -> detrMerged(outputs[0], spec, lb, names, threshold)
        OutputLayout.END2END -> end2end(outputs[0], spec, lb, names, threshold)
    }

    private fun label(names: List<String>, classId: Int) = names.getOrElse(classId) { "class $classId" }

    /** YOLOv8 / YOLO11: pixel box centre and size, then one probability per class (no objectness); NMS is applied here. */
    private fun yolov8(raw: FloatArray, spec: ModelSpec, lb: Letterbox, names: List<String>, threshold: Float, channelsFirst: Boolean): List<Detection> {
        val rows = spec.anchors
        val classes = spec.numClasses
        val k = 4 + classes
        val out = ArrayList<Detection>()
        val boxScale = if (spec.normalizedBoxes) spec.inputSize.toFloat() else 1f
        for (i in 0 until rows) {
            var best = -1
            var bestScore = threshold
            for (c in 0 until classes) {
                val s = if (channelsFirst) raw[(4 + c) * rows + i] else raw[i * k + 4 + c]
                if (s > bestScore) {
                    bestScore = s
                    best = c
                }
            }
            if (best < 0) continue
            val cx = (if (channelsFirst) raw[i] else raw[i * k]) * boxScale
            val cy = (if (channelsFirst) raw[rows + i] else raw[i * k + 1]) * boxScale
            val w = (if (channelsFirst) raw[2 * rows + i] else raw[i * k + 2]) * boxScale
            val h = (if (channelsFirst) raw[3 * rows + i] else raw[i * k + 3]) * boxScale
            YoloPostProcessor.toDetection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, bestScore, lb, label(names, best))?.let(out::add)
        }
        return YoloPostProcessor.nms(out, IOU)
    }

    /** DETR / RT-DETR / D-FINE with separate outputs: class logits (sigmoid scores) and normalised centre/size boxes. No NMS is needed. */
    private fun detrSplit(logits: FloatArray, boxes: FloatArray, spec: ModelSpec, lb: Letterbox, names: List<String>, threshold: Float): List<Detection> {
        val out = ArrayList<Detection>()
        val classes = spec.numClasses
        for (q in 0 until spec.queries) {
            var best = 0
            var bestLogit = logits[q * classes]
            for (c in 1 until classes) {
                val v = logits[q * classes + c]
                if (v > bestLogit) {
                    bestLogit = v
                    best = c
                }
            }
            val score = 1f / (1f + exp(-bestLogit))
            if (score <= threshold) continue
            addNormalisedBox(out, boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3], best, score, spec, lb, names)
        }
        return out
    }

    /** One output `[queries, 4 + classes]`: normalised centre/size, then class probabilities. */
    private fun detrMerged(raw: FloatArray, spec: ModelSpec, lb: Letterbox, names: List<String>, threshold: Float): List<Detection> {
        val out = ArrayList<Detection>()
        val classes = spec.numClasses
        val k = 4 + classes
        for (q in 0 until spec.queries) {
            var best = -1
            var bestScore = threshold
            for (c in 0 until classes) {
                val s = raw[q * k + 4 + c]
                if (s > bestScore) {
                    bestScore = s
                    best = c
                }
            }
            if (best < 0) continue
            addNormalisedBox(out, raw[q * k], raw[q * k + 1], raw[q * k + 2], raw[q * k + 3], best, bestScore, spec, lb, names)
        }
        return out
    }

    private fun addNormalisedBox(
        out: MutableList<Detection>, cx: Float, cy: Float, w: Float, h: Float, classId: Int, score: Float,
        spec: ModelSpec, lb: Letterbox, names: List<String>,
    ) {
        val s = spec.inputSize.toFloat()
        YoloPostProcessor.toDetection((cx - w / 2) * s, (cy - h / 2) * s, (cx + w / 2) * s, (cy + h / 2) * s, classId, score, lb, label(names, classId))?.let(out::add)
    }

    /** `[queries, 6]`: x1, y1, x2, y2, score, class. The corners are input pixels, or 0..1 fractions when none is above 1.5. */
    private fun end2end(raw: FloatArray, spec: ModelSpec, lb: Letterbox, names: List<String>, threshold: Float): List<Detection> {
        var maxCoordinate = 0f
        for (q in 0 until spec.queries) {
            if (raw[q * 6 + 4] <= threshold) continue
            for (c in 0 until 4) maxCoordinate = maxOf(maxCoordinate, raw[q * 6 + c])
        }
        val scale = if (maxCoordinate <= 1.5f) spec.inputSize.toFloat() else 1f
        val out = ArrayList<Detection>()
        for (q in 0 until spec.queries) {
            val score = raw[q * 6 + 4]
            if (score <= threshold) continue
            val classId = raw[q * 6 + 5].roundToInt()
            YoloPostProcessor.toDetection(
                raw[q * 6] * scale, raw[q * 6 + 1] * scale, raw[q * 6 + 2] * scale, raw[q * 6 + 3] * scale,
                classId, score, lb, label(names, classId),
            )?.let(out::add)
        }
        return out
    }
}

/** Puts the letterboxed ARGB pixels into a model input tensor in the order and value range the model expects. */
object InputPacker {
    /** HWC float (TFLite models from onnx2tf are NHWC). */
    fun nhwc(pixels: IntArray, convention: InputConvention, out: FloatArray) {
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            if (convention == InputConvention.BGR_255) {
                out[3 * i] = b.toFloat()
                out[3 * i + 1] = g.toFloat()
                out[3 * i + 2] = r.toFloat()
            } else {
                out[3 * i] = r / 255f
                out[3 * i + 1] = g / 255f
                out[3 * i + 2] = b / 255f
            }
        }
    }

    /** CHW float (ONNX models). */
    fun nchw(pixels: IntArray, convention: InputConvention, out: FloatBuffer) {
        val plane = pixels.size
        for (i in 0 until plane) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            if (convention == InputConvention.BGR_255) {
                out.put(i, b.toFloat())
                out.put(plane + i, g.toFloat())
                out.put(2 * plane + i, r.toFloat())
            } else {
                out.put(i, r / 255f)
                out.put(plane + i, g / 255f)
                out.put(2 * plane + i, b / 255f)
            }
        }
    }
}
