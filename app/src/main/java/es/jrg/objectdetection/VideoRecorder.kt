package es.jrg.objectdetection

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import java.io.File
import java.time.LocalDateTime

/**
 * Records the analysed frames, with their detection boxes drawn on, to an H.264 MP4.
 * Every method must be called from the same thread (the analyzer thread). Frames are timestamped when drawn, so the video plays
 * at the real detection rate. Saved to `Movies/ObjectDetection` (Android 10+) or the app's own Movies folder (older versions).
 */
class VideoRecorder(private val context: Context, private val width: Int, private val height: Int) {
    private val renderer = DetectionRenderer()
    private val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    private val inputSurface: Surface
    private val muxer: MediaMuxer
    private val target: Target
    private val info = MediaCodec.BufferInfo()
    private val startNs = System.nanoTime()
    private val dest = RectF(0f, 0f, width.toFloat(), height.toFloat())
    private var trackIndex = -1
    private var muxerStarted = false
    private var firstPtsUs = -1L
    private var frames = 0

    val elapsedMs: Long get() = (System.nanoTime() - startNs) / 1_000_000

    /** Where the video is written: a MediaStore entry (pending until finished) or a plain file. */
    private class Target(val uri: Uri?, val file: File?, val pfd: android.os.ParcelFileDescriptor?)

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, RecordingFormat.bitRate(width, height))
            setInteger(MediaFormat.KEY_FRAME_RATE, RecordingFormat.FRAME_RATE)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        inputSurface = codec.createInputSurface()
        codec.start()
        target = openTarget()
        Log.d("Inference", "recording started ${width}x$height")
        muxer = if (target.pfd != null) {
            MediaMuxer(target.pfd.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        } else {
            MediaMuxer(target.file!!.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        }
    }

    private fun openTarget(): Target {
        val name = RecordingFormat.fileName(LocalDateTime.now())
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, name)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/ObjectDetection")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
                ?: error("MediaStore insert failed")
            val pfd = context.contentResolver.openFileDescriptor(uri, "rw") ?: error("Cannot open $uri")
            return Target(uri, null, pfd)
        }
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: context.filesDir
        return Target(null, File(dir, name), null)
    }

    /** Draws [frame] and its [detections] onto the encoder input and writes out whatever the encoder has ready. */
    fun addFrame(frame: Bitmap, detections: List<Detection>) {
        val canvas = inputSurface.lockHardwareCanvas()
        try {
            canvas.drawBitmap(frame, null, dest, null)
            renderer.draw(canvas, detections, 0f, 0f, width.toFloat(), height.toFloat())
        } finally {
            inputSurface.unlockCanvasAndPost(canvas)
        }
        frames++
        drain(endOfStream = false)
    }

    /** Finishes the file. Returns true when a video with at least one frame was saved; otherwise the file is removed. */
    fun finish(): Boolean {
        var saved = false
        try {
            codec.signalEndOfInputStream()
            drain(endOfStream = true)
            saved = muxerStarted && frames > 0
        } catch (e: Exception) {
            Log.w("Inference", "Finishing the recording failed", e)
        } finally {
            runCatching { codec.stop() }
            codec.release()
            inputSurface.release()
            if (muxerStarted) runCatching { muxer.stop() }
            muxer.release()
            target.pfd?.close()
            publish(saved)
        }
        return saved
    }

    private fun publish(saved: Boolean) {
        if (target.uri != null) {
            if (saved) {
                val values = ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }
                context.contentResolver.update(target.uri, values, null, null)
            } else {
                context.contentResolver.delete(target.uri, null, null)
            }
        } else if (!saved) {
            target.file?.delete()
        }
    }

    private fun drain(endOfStream: Boolean) {
        var idleTries = 0
        while (true) {
            val index = codec.dequeueOutputBuffer(info, if (endOfStream) 10_000L else 0L)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream || ++idleTries > 200) return
                }
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    trackIndex = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    muxerStarted = true
                }
                index >= 0 -> {
                    val buffer = codec.getOutputBuffer(index)
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                    if (buffer != null && info.size > 0 && muxerStarted) {
                        if (firstPtsUs < 0) firstPtsUs = info.presentationTimeUs
                        info.presentationTimeUs -= firstPtsUs
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        muxer.writeSampleData(trackIndex, buffer, info)
                    }
                    codec.releaseOutputBuffer(index, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }
}
