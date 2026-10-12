package com.example.objectdetection

import kotlin.math.max
import kotlin.math.min

/** A detection with a box in normalized (0..1) coordinates of the source image. */
data class Detection(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val classId: Int,
    val score: Float,
) {
    val label: String get() = CocoLabels.name(classId)
}

/** Maps between a source image and the square, gray-padded model input (YOLO "letterbox"). */
class Letterbox(val srcWidth: Int, val srcHeight: Int, val size: Int) {
    val scale: Float = min(size.toFloat() / srcWidth, size.toFloat() / srcHeight)
    val padX: Float = (size - srcWidth * scale) / 2f
    val padY: Float = (size - srcHeight * scale) / 2f
}

/** Helpers shared by the YOLO-style decoders: box mapping, non-maximum suppression and IoU. */
object YoloPostProcessor {
    const val NUM_CLASSES = 80

    /** Model-space corners -> normalized corners of the source image (undo padding and scale). */
    internal fun toDetection(l: Float, t: Float, r: Float, b: Float, classId: Int, score: Float, lb: Letterbox): Detection? {
        val left = ((l - lb.padX) / lb.scale / lb.srcWidth).coerceIn(0f, 1f)
        val right = ((r - lb.padX) / lb.scale / lb.srcWidth).coerceIn(0f, 1f)
        val top = ((t - lb.padY) / lb.scale / lb.srcHeight).coerceIn(0f, 1f)
        val bottom = ((b - lb.padY) / lb.scale / lb.srcHeight).coerceIn(0f, 1f)
        return if (right > left && bottom > top) Detection(left, top, right, bottom, classId, score) else null
    }

    /** Class-aware non-maximum suppression. */
    fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val kept = ArrayList<Detection>()
        for (d in sorted) {
            if (kept.none { it.classId == d.classId && iou(it, d) > iouThreshold }) kept.add(d)
        }
        return kept
    }

    fun iou(a: Detection, b: Detection): Float {
        val iw = min(a.right, b.right) - max(a.left, b.left)
        val ih = min(a.bottom, b.bottom) - max(a.top, b.top)
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val union = area(a) + area(b) - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun area(d: Detection) = (d.right - d.left) * (d.bottom - d.top)
}
