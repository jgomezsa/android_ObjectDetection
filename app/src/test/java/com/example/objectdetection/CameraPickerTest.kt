package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Test

class CameraPickerTest {
    @Test
    fun prefersTheCameraThatZoomsOutTheFurthest() {
        // The Poco X7 Pro: id 0 main (listed), id 2 ultrawide physical (hidden), id 3 logical 0.6-10 (hidden), id 4 logical 1-10 (hidden).
        val ordered = CameraPicker.order(
            listOf(
                CameraCandidate("0", true, 1f, 10f),
                CameraCandidate("2", false, 1f, 4f),
                CameraCandidate("3", false, 0.6f, 10f),
                CameraCandidate("4", false, 1f, 10f),
            ),
        )
        assertEquals(listOf("3", "0", "4", "2"), ordered.map { it.id })
    }

    @Test
    fun withoutAnUltrawideTheListedMainCameraComesFirst() {
        val ordered = CameraPicker.order(listOf(CameraCandidate("5", false, 1f, 10f), CameraCandidate("0", true, 1f, 8f)))
        assertEquals(listOf("0", "5"), ordered.map { it.id })
    }

    @Test
    fun camerasThatCannotReachOneXAreSkipped() {
        val ordered = CameraPicker.order(listOf(CameraCandidate("7", false, 2f, 5f), CameraCandidate("0", true, 1f, 10f)))
        assertEquals(listOf("0"), ordered.map { it.id })
    }
}
