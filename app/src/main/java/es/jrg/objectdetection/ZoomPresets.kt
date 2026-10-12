package es.jrg.objectdetection

import kotlin.math.ceil
import kotlin.math.roundToInt

/** Zoom stops offered to the user, derived from what the camera reports so every device gets only the stops it supports. */
object ZoomPresets {
    /** Zoom beyond 10x is only a blurry crop, so the ladder stops there. */
    private val LADDER = listOf(2f, 5f, 10f)

    /**
     * The camera's own minimum when it is below 1x (an ultrawide lens), 1x when in range, then the ladder values the camera can reach.
     * Returns an empty list when fewer than two stops exist (nothing to choose from).
     */
    fun forRange(min: Float, max: Float): List<Float> {
        val points = mutableListOf<Float>()
        // Round the minimum up to one decimal (0.60000002 -> 0.6) so the stop never falls below the real minimum.
        if (min < 0.95f) points += ceil(min * 10 - 1e-3f) / 10f
        if (min <= 1f && max >= 1f) points += 1f
        points += LADDER.filter { it >= min && it <= max + 1e-3f }
        return points.distinct().takeIf { it.size >= 2 } ?: emptyList()
    }

    /** Position (0..1) of [ratio] along a selector whose [points] are evenly spaced; between two stops it is interpolated linearly. */
    fun position(points: List<Float>, ratio: Float): Float {
        if (points.size < 2 || ratio <= points.first()) return 0f
        if (ratio >= points.last()) return 1f
        val i = points.indexOfLast { it <= ratio }
        val t = (ratio - points[i]) / (points[i + 1] - points[i])
        return (i + t) / (points.size - 1)
    }

    /** Inverse of [position]. */
    fun ratioAt(points: List<Float>, position: Float): Float {
        val p = position.coerceIn(0f, 1f) * (points.size - 1)
        val i = p.toInt().coerceAtMost(points.size - 2)
        return points[i] + (p - i) * (points[i + 1] - points[i])
    }

    /** The stop closest to a selector [position] (0..1). */
    fun nearestStop(points: List<Float>, position: Float): Float =
        points[(position.coerceIn(0f, 1f) * (points.size - 1)).roundToInt()]

    /** "0.6x", "1x", "2x". */
    fun label(ratio: Float): String =
        if (ratio == ratio.roundToInt().toFloat()) "${ratio.roundToInt()}x" else "${ratio}x"
}
