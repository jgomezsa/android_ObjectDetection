package com.example.objectdetection

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

/** Draws detection boxes and labels on a native canvas; shared by the on-screen overlay and the video recorder. */
class DetectionRenderer {
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val bgPaint = Paint().apply { style = Paint.Style.FILL }

    /** Detections use coordinates normalised to the frame, which occupies the rectangle at ([originX], [originY]) of size [frameW] x [frameH]. */
    fun draw(canvas: Canvas, detections: List<Detection>, originX: Float, originY: Float, frameW: Float, frameH: Float) {
        val unit = frameW / 1000f
        boxPaint.strokeWidth = 3.3f * unit
        textPaint.textSize = 34.4f * unit
        val labelH = 42.6f * unit
        val labelPad = 6.6f * unit
        val baselineGap = 9.8f * unit
        for (d in detections) {
            val color = Color.HSVToColor(floatArrayOf((d.classId * 47 % 360).toFloat(), 0.85f, 1f))
            val left = originX + d.left * frameW
            val top = originY + d.top * frameH
            boxPaint.color = color
            canvas.drawRect(left, top, originX + d.right * frameW, originY + d.bottom * frameH, boxPaint)
            val text = "${d.label} ${(d.score * 100).toInt()}%"
            // Keep the label inside the frame when the box touches its top edge.
            val labelTop = maxOf(top - labelH, originY)
            bgPaint.color = (color and 0x00FFFFFF) or (0xCC shl 24)
            canvas.drawRect(left, labelTop, left + textPaint.measureText(text) + 2 * labelPad, labelTop + labelH, bgPaint)
            canvas.drawText(text, left + labelPad, labelTop + labelH - baselineGap, textPaint)
        }
    }
}
