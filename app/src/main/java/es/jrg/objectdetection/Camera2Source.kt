package es.jrg.objectdetection

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.WindowManager
import kotlin.math.abs

/** Camera analysis size (landscape 4:3), also what is shown on screen and recorded, so it sets the picture sharpness. The closest supported size is used. */
private const val ANALYSIS_WIDTH = 1280
private const val ANALYSIS_HEIGHT = 960
private const val TAG = "Camera2"

/**
 * Camera2 frame source. It looks for the back camera that zooms out the furthest (on some phones a logical camera that includes
 * the ultrawide lens is not listed publicly but can be opened, which CameraX cannot do), opens it, and delivers upright ARGB
 * bitmaps through [onFrame] on its own image thread. If that camera cannot be opened it tries the next best one.
 *
 * Frames are handed over one at a time: while [onFrame] runs, newer frames replace each other in the [ImageReader] and only the
 * latest one is taken afterwards.
 */
@SuppressLint("MissingPermission") // the screen only starts the source once the CAMERA permission is granted
class Camera2Source(
    private val context: Context,
    /** Called when a camera is streaming, with its zoom range. */
    private val onCameraReady: (zoomMin: Float, zoomMax: Float) -> Unit,
    /** Called for each frame with the upright bitmap and the [System.nanoTime] at which the frame started being processed. */
    private val onFrame: (Bitmap, Long) -> Unit,
    private val onError: (String) -> Unit,
) {
    private class Chosen(
        val id: String,
        val characteristics: CameraCharacteristics,
        val candidate: CameraCandidate,
        val size: Size,
        val sensorOrientation: Int,
    )

    private val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private var controlThread: HandlerThread? = null
    private var imageThread: HandlerThread? = null
    private var control: Handler? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var builder: CaptureRequest.Builder? = null
    private var active: Chosen? = null

    @Volatile private var zoom = 1f

    /** Incremented by every start() and stop(), so callbacks of an older session are ignored. */
    @Volatile private var generation = 0

    // Reused per frame (the image thread is the only user).
    private var yBytes = ByteArray(0)
    private var uBytes = ByteArray(0)
    private var vBytes = ByteArray(0)
    private var pixels = IntArray(0)

    fun start() {
        if (control != null) return
        val gen = ++generation
        val controlThread = HandlerThread("camera-control").also { it.start() }
        imageThread = HandlerThread("camera-image").also { it.start() }
        this.controlThread = controlThread
        val handler = Handler(controlThread.looper)
        control = handler
        handler.post { open(discover(), 0, gen) }
    }

    /** Closes the camera. With [wait], returns only after the image thread has finished the frame it was processing. */
    fun stop(wait: Boolean = false) {
        val handler = control ?: return
        generation++
        val closingControl = controlThread
        val closingImage = imageThread
        control = null
        controlThread = null
        imageThread = null
        handler.post {
            runCatching { session?.close() }
            runCatching { device?.close() }
            runCatching { reader?.close() }
            session = null
            device = null
            reader = null
            builder = null
            active = null
            closingControl?.quitSafely()
        }
        closingImage?.quitSafely()
        if (wait) {
            closingImage?.join(3000)
            closingControl?.join(1000)
        }
    }

    fun setZoom(ratio: Float) {
        zoom = ratio
        control?.post {
            val request = builder ?: return@post
            val chosen = active ?: return@post
            applyZoom(request, chosen)
            try {
                session?.setRepeatingRequest(request.build(), null, control)
            } catch (e: Exception) {
                Log.w(TAG, "Could not change the zoom", e)
            }
        }
    }

    /** Back cameras that can be opened, best first. Besides the public list, a few more ids are probed: some phones hide their ultrawide camera there. */
    private fun discover(): List<Chosen> {
        val listed = manager.cameraIdList.toList()
        val ids = (listed + (0..9).map { it.toString() }).distinct()
        val all = ids.mapNotNull { describe(it, it in listed) }
        all.forEach { Log.d(TAG, "camera ${it.id} listed=${it.candidate.listed} zoom ${it.candidate.zoomMin}..${it.candidate.zoomMax} size ${it.size}") }
        val order = CameraPicker.order(all.map { it.candidate }).map { it.id }
        return order.mapNotNull { id -> all.firstOrNull { it.id == id } }
    }

    private fun describe(id: String, listed: Boolean): Chosen? = try {
        val c = manager.getCameraCharacteristics(id)
        val capabilities = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toList().orEmpty()
        val sizes = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)?.getOutputSizes(ImageFormat.YUV_420_888)
        if (c.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK ||
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE !in capabilities ||
            sizes.isNullOrEmpty()
        ) {
            null
        } else {
            val zoomRange = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) c.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) else null
            val min = zoomRange?.lower ?: 1f
            val max = zoomRange?.upper ?: (c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f)
            Chosen(id, c, CameraCandidate(id, listed, min, max), pickSize(sizes), c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90)
        }
    } catch (e: Exception) {
        // Unknown or system-only ids throw; that just means this id is not usable.
        null
    }

    private fun pickSize(sizes: Array<Size>): Size {
        val target = ANALYSIS_WIDTH.toLong() * ANALYSIS_HEIGHT
        val fourByThree = sizes.filter { it.width >= it.height && abs(it.width.toFloat() / it.height - 4f / 3f) < 0.02f }
        return (fourByThree.ifEmpty { sizes.toList() }).minBy { abs(it.width.toLong() * it.height - target) }
    }

    private fun open(candidates: List<Chosen>, index: Int, gen: Int) {
        if (gen != generation) return
        if (index >= candidates.size) {
            onError("Could not open any camera")
            return
        }
        val chosen = candidates[index]
        fun tryNext() = open(candidates, index + 1, gen)
        try {
            manager.openCamera(
                chosen.id,
                object : CameraDevice.StateCallback() {
                    override fun onOpened(camera: CameraDevice) {
                        if (gen != generation) {
                            camera.close()
                            return
                        }
                        device = camera
                        startSession(camera, chosen, gen) {
                            camera.close()
                            if (device === camera) device = null
                            tryNext()
                        }
                    }

                    override fun onDisconnected(camera: CameraDevice) {
                        camera.close()
                        if (device === camera) device = null
                        if (gen == generation && session == null) tryNext()
                    }

                    override fun onError(camera: CameraDevice, error: Int) {
                        Log.w(TAG, "camera ${chosen.id} error $error")
                        camera.close()
                        if (device === camera) device = null
                        if (gen == generation && session == null) tryNext()
                    }
                },
                control,
            )
        } catch (e: Exception) {
            Log.w(TAG, "camera ${chosen.id} could not be opened", e)
            tryNext()
        }
    }

    @Suppress("DEPRECATION") // createCaptureSession(List, ...) works on every supported API level
    private fun startSession(camera: CameraDevice, chosen: Chosen, gen: Int, onFailed: () -> Unit) {
        val imageReader = ImageReader.newInstance(chosen.size.width, chosen.size.height, ImageFormat.YUV_420_888, 3)
        imageReader.setOnImageAvailableListener({ handleImage(it, chosen) }, imageThread?.looper?.let { Handler(it) })
        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(imageReader.surface)
            val afModes = chosen.characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
            if (afModes != null && CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE in afModes) {
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            }
        }
        applyZoom(request, chosen)
        try {
            camera.createCaptureSession(
                listOf(imageReader.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(captureSession: CameraCaptureSession) {
                        if (gen != generation) {
                            captureSession.close()
                            imageReader.close()
                            return
                        }
                        session = captureSession
                        reader = imageReader
                        builder = request
                        active = chosen
                        try {
                            captureSession.setRepeatingRequest(request.build(), null, control)
                        } catch (e: CameraAccessException) {
                            Log.w(TAG, "camera ${chosen.id} refused the request", e)
                            session = null
                            imageReader.close()
                            onFailed()
                            return
                        }
                        Log.d(TAG, "streaming camera ${chosen.id} at ${chosen.size}, zoom ${chosen.candidate.zoomMin}..${chosen.candidate.zoomMax}")
                        onCameraReady(chosen.candidate.zoomMin, chosen.candidate.zoomMax)
                    }

                    override fun onConfigureFailed(captureSession: CameraCaptureSession) {
                        Log.w(TAG, "camera ${chosen.id} session configuration failed")
                        captureSession.close()
                        imageReader.close()
                        onFailed()
                    }
                },
                control,
            )
        } catch (e: Exception) {
            Log.w(TAG, "camera ${chosen.id} session could not be created", e)
            imageReader.close()
            onFailed()
        }
    }

    private fun applyZoom(request: CaptureRequest.Builder, chosen: Chosen) {
        val ratio = zoom.coerceIn(chosen.candidate.zoomMin, chosen.candidate.zoomMax)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            request.set(CaptureRequest.CONTROL_ZOOM_RATIO, ratio)
        } else {
            val sensor = chosen.characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return
            val w = (sensor.width() / ratio).toInt()
            val h = (sensor.height() / ratio).toInt()
            val left = sensor.centerX() - w / 2
            val top = sensor.centerY() - h / 2
            request.set(CaptureRequest.SCALER_CROP_REGION, Rect(left, top, left + w, top + h))
        }
    }

    private fun handleImage(imageReader: ImageReader, chosen: Chosen) {
        val image = try {
            imageReader.acquireLatestImage()
        } catch (e: IllegalStateException) {
            null // the reader was closed
        } ?: return
        try {
            val started = System.nanoTime()
            onFrame(toBitmap(image, chosen), started)
        } catch (e: IllegalStateException) {
            Log.w(TAG, "Frame dropped (camera closing?)", e)
        } finally {
            image.close()
        }
    }

    private fun toBitmap(image: Image, chosen: Chosen): Bitmap {
        val width = image.width
        val height = image.height
        val planes = image.planes
        yBytes = copy(planes[0], yBytes)
        uBytes = copy(planes[1], uBytes)
        vBytes = copy(planes[2], vBytes)
        if (pixels.size < width * height) pixels = IntArray(width * height)
        val rotation = rotationDegrees(chosen.sensorOrientation)
        YuvConverter.toArgb(
            yBytes, planes[0].rowStride,
            uBytes, vBytes, planes[1].rowStride, planes[1].pixelStride,
            width, height, rotation, pixels,
        )
        val rotated = rotation == 90 || rotation == 270
        return Bitmap.createBitmap(pixels, if (rotated) height else width, if (rotated) width else height, Bitmap.Config.ARGB_8888)
    }

    private fun copy(plane: Image.Plane, into: ByteArray): ByteArray {
        val buffer = plane.buffer
        buffer.rewind()
        val target = if (into.size >= buffer.remaining()) into else ByteArray(buffer.remaining())
        buffer.get(target, 0, buffer.remaining())
        return target
    }

    /** Clockwise rotation that makes the back camera's frame upright for the current screen rotation. */
    @Suppress("DEPRECATION")
    private fun rotationDegrees(sensorOrientation: Int): Int {
        val display = (context.getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay.rotation
        val screen = when (display) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (sensorOrientation - screen + 360) % 360
    }
}
