package com.example.objectdetection

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/** Pure helpers for video recording (kept free of Android classes so they test on the JVM). */
object RecordingFormat {
    const val FRAME_RATE = 15
    private const val BITS_PER_PIXEL = 4

    /** H.264 encoders need even dimensions. */
    fun evenSize(size: Int): Int = size and 1.inv()

    fun bitRate(width: Int, height: Int): Int = width * height * BITS_PER_PIXEL

    fun fileName(time: LocalDateTime): String =
        "ObjectDetection_${time.format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))}.mp4"

    /** Elapsed recording time as m:ss. */
    fun elapsed(ms: Long): String {
        val seconds = ms / 1000
        return "${seconds / 60}:${(seconds % 60).toString().padStart(2, '0')}"
    }
}
