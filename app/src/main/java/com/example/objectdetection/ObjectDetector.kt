package com.example.objectdetection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF

data class DetectionResult(
    val detections: List<Detection>,
    /** The whole detect() call; the rest is its breakdown (preprocess / model run / decode+NMS). */
    val inferenceMs: Long,
    val preprocessMs: Long = 0,
    val runMs: Long = 0,
    val postMs: Long = 0,
)

/** A loaded model that turns upright camera frames into detections. */
interface ObjectDetector : AutoCloseable {
    /** Backend the model is running on (the GPU, or the CPU when the GPU could not run it). */
    val backend: Backend

    /** [bitmap] must already be upright (rotation applied). Boxes are normalized to it. */
    fun detect(bitmap: Bitmap): DetectionResult
}

/** Draws a frame into a square, gray-padded model input (YOLO "letterbox") and returns its ARGB pixels. */
class LetterboxRenderer(private val size: Int) {
    private val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(bitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    val pixels = IntArray(size * size)

    fun render(source: Bitmap, lb: Letterbox): IntArray {
        canvas.drawColor(Color.rgb(114, 114, 114))
        val dst = RectF(lb.padX, lb.padY, lb.padX + source.width * lb.scale, lb.padY + source.height * lb.scale)
        canvas.drawBitmap(source, null, dst, paint)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        return pixels
    }
}
