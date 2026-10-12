package es.jrg.objectdetection

import java.util.stream.IntStream

/** Converts a camera YUV_420_888 frame to upright ARGB pixels (JPEG-style full-range BT.601), splitting the rows over the CPU cores. */
object YuvConverter {
    private const val BANDS = 8

    /**
     * [y], [u], [v] are the plane contents with their row strides ([uvRowStride], [uvPixelStride] apply to both chroma planes).
     * [rotation] (0, 90, 180 or 270 degrees clockwise) is applied while converting; [out] receives `width * height` pixels laid out
     * for the rotated image (so its width is `height` when rotated by 90 or 270).
     */
    fun toArgb(
        y: ByteArray, yRowStride: Int,
        u: ByteArray, v: ByteArray, uvRowStride: Int, uvPixelStride: Int,
        width: Int, height: Int, rotation: Int, out: IntArray,
    ) {
        require(out.size >= width * height) { "Output too small" }
        val rowsPerBand = (height + BANDS - 1) / BANDS
        IntStream.range(0, BANDS).parallel().forEach { band ->
            val rowStart = band * rowsPerBand
            val rowEnd = minOf(height, rowStart + rowsPerBand)
            for (row in rowStart until rowEnd) {
                val yBase = row * yRowStride
                val uvBase = (row shr 1) * uvRowStride
                for (col in 0 until width) {
                    val yy = y[yBase + col].toInt() and 0xFF
                    val uvIndex = uvBase + (col shr 1) * uvPixelStride
                    val uu = (u[uvIndex].toInt() and 0xFF) - 128
                    val vv = (v[uvIndex].toInt() and 0xFF) - 128
                    val r = clamp(yy + ((91881 * vv) shr 16))
                    val g = clamp(yy - ((22554 * uu + 46802 * vv) shr 16))
                    val b = clamp(yy + ((116130 * uu) shr 16))
                    out[destination(col, row, width, height, rotation)] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
    }

    /** Index in the rotated image of the source pixel ([x], [y]). */
    fun destination(x: Int, y: Int, width: Int, height: Int, rotation: Int): Int = when (rotation) {
        90 -> x * height + (height - 1 - y)
        180 -> (height - 1 - y) * width + (width - 1 - x)
        270 -> (width - 1 - x) * height + y
        else -> y * width + x
    }

    private fun clamp(value: Int) = if (value < 0) 0 else if (value > 255) 255 else value
}
