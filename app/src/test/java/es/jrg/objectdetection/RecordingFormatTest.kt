package es.jrg.objectdetection

import java.time.LocalDateTime
import org.junit.Assert.assertEquals
import org.junit.Test

class RecordingFormatTest {
    @Test
    fun evenSizeRoundsDownToEven() {
        assertEquals(960, RecordingFormat.evenSize(960))
        assertEquals(960, RecordingFormat.evenSize(961))
    }

    @Test
    fun bitRateScalesWithPixels() {
        assertEquals(960 * 1280 * 4, RecordingFormat.bitRate(960, 1280))
    }

    @Test
    fun fileNameUsesTimestamp() {
        assertEquals(
            "ObjectDetection_20261012_185507.mp4",
            RecordingFormat.fileName(LocalDateTime.of(2026, 10, 12, 18, 55, 7)),
        )
    }

    @Test
    fun elapsedIsMinutesAndPaddedSeconds() {
        assertEquals("0:00", RecordingFormat.elapsed(0))
        assertEquals("0:07", RecordingFormat.elapsed(7_499))
        assertEquals("1:05", RecordingFormat.elapsed(65_000))
        assertEquals("12:00", RecordingFormat.elapsed(720_000))
    }
}
