package es.jrg.objectdetection

/** A back camera the app can open, with the zoom range it reports. [listed] is false for cameras Android does not list publicly (found by probing). */
data class CameraCandidate(val id: String, val listed: Boolean, val zoomMin: Float, val zoomMax: Float)

object CameraPicker {
    /**
     * Cameras in the order they should be tried: the one that zooms out the furthest first (a logical camera that includes the
     * ultrawide lens, where the phone allows it), then publicly listed cameras before hidden ones, then the larger zoom range.
     * Cameras that cannot reach 1x are skipped (they are not a normal wide camera).
     */
    fun order(candidates: List<CameraCandidate>): List<CameraCandidate> =
        candidates
            .filter { it.zoomMin <= 1f + 1e-3f }
            .sortedWith(compareBy<CameraCandidate> { it.zoomMin }.thenBy { !it.listed }.thenByDescending { it.zoomMax })
}
