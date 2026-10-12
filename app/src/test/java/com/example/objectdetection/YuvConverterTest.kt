package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YuvConverterTest {
    private fun channel(argb: Int, shift: Int) = (argb shr shift) and 0xFF

    @Test
    fun grayStaysGrayAndOpaque() {
        val out = IntArray(4)
        YuvConverter.toArgb(ByteArray(4) { 100 }, 2, ByteArray(1) { 128.toByte() }, ByteArray(1) { 128.toByte() }, 1, 1, 2, 2, 0, out)
        out.forEach {
            assertEquals(0xFF, channel(it, 24))
            assertEquals(100, channel(it, 16))
            assertEquals(100, channel(it, 8))
            assertEquals(100, channel(it, 0))
        }
    }

    @Test
    fun redChromaGivesARedPixel() {
        // Full-range BT.601 red is Y=76, U=85, V=255.
        val out = IntArray(4)
        YuvConverter.toArgb(ByteArray(4) { 76 }, 2, ByteArray(1) { 85 }, ByteArray(1) { 255.toByte() }, 1, 1, 2, 2, 0, out)
        assertTrue(channel(out[0], 16) > 240)
        assertTrue(channel(out[0], 8) < 15)
        assertTrue(channel(out[0], 0) < 15)
    }

    @Test
    fun interleavedChromaPlanesUsePixelStrideTwo() {
        // Semi-planar layout: U and V alternate, so pixel stride is 2 and the V plane starts one byte after the U plane.
        val uv = byteArrayOf(85, 255.toByte(), 85, 255.toByte())
        val out = IntArray(8)
        YuvConverter.toArgb(ByteArray(8) { 76 }, 4, uv, uv.copyOfRange(1, 4), 4, 2, 4, 2, 0, out)
        assertTrue(channel(out[3], 16) > 240)
    }

    @Test
    fun rotationMovesPixelsLikeAClockwiseTurn() {
        // 4x2 image whose pixel value encodes its position: Y(x, y) = 20 + 30 * x + 100 * y.
        val width = 4
        val height = 2
        val y = ByteArray(width * height) { (20 + 30 * (it % width) + 100 * (it / width)).toByte() }
        val gray = ByteArray(2) { 128.toByte() }
        for (rotation in listOf(0, 90, 180, 270)) {
            val out = IntArray(width * height)
            YuvConverter.toArgb(y, width, gray, gray, 2, 1, width, height, rotation, out)
            for (py in 0 until height) {
                for (px in 0 until width) {
                    val expected = 20 + 30 * px + 100 * py
                    assertEquals("rotation $rotation at ($px,$py)", expected, channel(out[YuvConverter.destination(px, py, width, height, rotation)], 0))
                }
            }
        }
        // 90 degrees clockwise: the top-left source pixel ends up at the top-right of the 2-wide result.
        assertEquals(1, YuvConverter.destination(0, 0, width, height, 90))
        // The top-right source pixel (3, 0) ends up at the bottom-right.
        assertEquals(3 * 2 + 1, YuvConverter.destination(3, 0, width, height, 90))
    }
}
