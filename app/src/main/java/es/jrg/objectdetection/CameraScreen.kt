package es.jrg.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.min
import kotlin.math.roundToInt

/** Number of recent frames shown in the processing-time plot. */
private const val FRAME_HISTORY = 50

/** Range of the confidence slider (1% steps). */
private const val THRESHOLD_MIN = 0.01f
private const val THRESHOLD_MAX = 0.99f

/**
 * Loads [model] on the first backend that can run it: for a TFLite model the GPU, then the CPU (or only the one a developer forces);
 * an ONNX model runs on the CPU only (ONNX Runtime has no GPU backend here). The user chose the model, so a slow load is accepted.
 */
private fun createDetector(context: Context, model: ModelInfo, backends: List<Backend>): ObjectDetector = when (model.format) {
    ModelFormat.TFLITE -> BackendChain.firstWorking(backends) { LiteRtDetector(context, model, it) }.second
    ModelFormat.ONNX -> BackendChain.firstWorking(backends.filter { it == Backend.CPU }) { OrtDetector(context, model, it) }.second
}

/**
 * [initialModel] / [forcedBackend] are developer overrides (launch extras `model` / `backend`):
 * `adb shell am start -n es.jrg.objectdetection/.MainActivity --es model yolox_tiny --es backend CPU`.
 */
@Composable
fun CameraScreen(
    contentPadding: PaddingValues = PaddingValues(),
    initialModel: String? = null,
    forcedBackend: Backend? = null,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val prefs = remember { context.getSharedPreferences("settings", Context.MODE_PRIVATE) }
    // Built-in models plus the ones the user downloaded (re-read after adding or deleting).
    val store = remember { ModelStore(File(context.filesDir, "models")) }
    val installer = remember { ModelInstaller(context, store) }
    var models by remember { mutableStateOf(store.all()) }
    var showManage by remember { mutableStateOf(false) }
    var showAbout by remember { mutableStateOf(false) }
    val startModel = remember { initialModel ?: prefs.getString("model", null)?.takeIf { id -> models.any { it.id == id } } ?: DEFAULT_MODEL }
    val backends = remember { forcedBackend?.let { listOf(it) } ?: BackendChain.DEFAULT_ORDER }
    // Minimum score for a box to be shown; the analyzer thread reads the atomic copy.
    val savedThreshold = remember { prefs.getFloat("threshold", OutputDecoders.DEFAULT_THRESHOLD).coerceIn(THRESHOLD_MIN, THRESHOLD_MAX) }
    var threshold by remember { mutableStateOf(savedThreshold) }
    val thresholdRef = remember { AtomicReference(savedThreshold) }
    var result by remember { mutableStateOf(DetectionResult(emptyList(), 0)) }
    // The frame on screen together with the detections computed on that very frame, so boxes never lag behind the picture.
    var shown by remember { mutableStateOf<ShownFrame?>(null) }
    // The model the user picked; the analyzer thread (re)loads the detector when it differs from the loaded one.
    var selectedModel by remember { mutableStateOf(startModel) }
    val requestedModel = remember { AtomicReference(startModel) }
    // A model that failed to load is not retried on every frame; picking a model again clears this.
    val failedModel = remember { AtomicReference("") }
    var loading by remember { mutableStateOf(true) }
    var loadFailed by remember { mutableStateOf(false) }
    var usingGpu by remember { mutableStateOf(false) }
    var frameTimes by remember { mutableStateOf(emptyList<Long>()) }
    // When false the camera image keeps updating but frames are not analysed (no detection, no boxes).
    var processing by remember { mutableStateOf(true) }
    val processingFlag = remember { AtomicBoolean(true) }
    // Video recording: the UI sets the flag, the analyzer thread starts/stops the recorder and reports the elapsed time.
    var recording by remember { mutableStateOf(false) }
    var recordedMs by remember { mutableStateOf(0L) }
    val recordFlag = remember { AtomicBoolean(false) }
    // Zoom: the stops this camera offers (empty = no zoom control) and the current ratio. Plain State objects so gesture handlers always see the latest values.
    val sourceRef = remember { AtomicReference<Camera2Source?>(null) }
    val zoomPoints = remember { mutableStateOf(emptyList<Float>()) }
    val zoomRatio = remember { mutableStateOf(1f) }
    val setZoom = remember {
        { ratio: Float ->
            val points = zoomPoints.value
            val source = sourceRef.get()
            if (source != null && points.size >= 2) {
                val target = ratio.coerceIn(points.first(), points.last())
                zoomRatio.value = target
                source.setZoom(target)
            }
        }
    }
    val selectModel = remember {
        { model: String ->
            failedModel.set("")
            selectedModel = model
            requestedModel.set(model)
            prefs.edit().putString("model", model).apply()
        }
    }

    DisposableEffect(lifecycleOwner) {
        val history = FrameTimeHistory(FRAME_HISTORY)
        var detector: ObjectDetector? = null
        var loadedModel = ""
        var recorder: VideoRecorder? = null
        val mainExecutor = ContextCompat.getMainExecutor(context)
        fun toast(message: String) = mainExecutor.execute { Toast.makeText(context, message, Toast.LENGTH_LONG).show() }
        fun stopRecording(announce: Boolean = true) {
            val rec = recorder ?: return
            recorder = null
            recordFlag.set(false)
            recording = false
            val saved = rec.finish()
            Log.d("Inference", "recording finished, saved=$saved")
            if (announce) toast(if (saved) "Video saved to Movies/ObjectDetection" else "Nothing was recorded")
        }

        // Runs on the camera's image thread, one frame at a time.
        fun onFrame(upright: Bitmap, frameStart: Long) {
            if (!processingFlag.get()) {
                // Paused: keep the picture live, without boxes.
                stopRecording()
                shown = ShownFrame(upright, emptyList())
                return
            }
            val wanted = requestedModel.get()
            if (wanted != loadedModel && wanted != failedModel.get()) {
                val wantedInfo = store.find(wanted)
                // Loading can take many seconds (GPU compilation); show the picture meanwhile.
                shown = ShownFrame(upright, emptyList())
                loading = true
                val loadStart = System.nanoTime()
                detector?.close()
                detector = null
                val previous = loadedModel
                loadedModel = ""
                try {
                    if (wantedInfo == null) throw ModelRejectedException("The model $wanted is not in the list any more.")
                    val loaded = createDetector(context, wantedInfo, backends)
                    detector = loaded
                    loadedModel = wanted
                    usingGpu = loaded.backend.isGpu
                    history.clear()
                    loadFailed = false
                    loading = false
                    Log.d("Inference", "loaded $wanted ${loaded.backend.name} in ${(System.nanoTime() - loadStart) / 1_000_000} ms")
                } catch (e: Throwable) {
                    Log.e("Inference", "Could not load $wanted", e)
                    failedModel.set(wanted)
                    toast("Could not load ${wantedInfo?.displayName ?: wanted}")
                    if (previous.isNotEmpty()) {
                        // Go back to the model that was working; it is reloaded on the next frame.
                        selectedModel = previous
                        requestedModel.set(previous)
                    } else {
                        loadFailed = true
                    }
                    return
                }
            }
            val active = detector
            if (active == null) {
                shown = ShownFrame(upright, emptyList())
                return
            }
            val r = active.detect(upright, thresholdRef.get())
            val frameMs = (System.nanoTime() - frameStart) / 1_000_000
            Log.d(
                "Inference",
                "$loadedModel ${active.backend.name} ${r.inferenceMs} ms " +
                    "(pre ${r.preprocessMs}, run ${r.runMs}, post ${r.postMs}) frame $frameMs ms",
            )
            shown = ShownFrame(upright, r.detections)
            result = r
            history.add(r.inferenceMs)
            frameTimes = history.snapshot()
            if (recordFlag.get()) {
                try {
                    val rec = recorder ?: VideoRecorder(
                        context,
                        RecordingFormat.evenSize(upright.width),
                        RecordingFormat.evenSize(upright.height),
                    ).also { recorder = it }
                    rec.addFrame(upright, r.detections)
                    recordedMs = rec.elapsedMs
                } catch (e: Exception) {
                    Log.e("Inference", "Recording failed", e)
                    runCatching { stopRecording(announce = false) }
                    recordFlag.set(false)
                    recording = false
                    toast("Recording failed")
                }
            } else {
                stopRecording()
            }
        }

        val source = Camera2Source(
            context,
            onCameraReady = { min, max ->
                val points = ZoomPresets.forRange(min, max)
                zoomPoints.value = points
                zoomRatio.value = if (points.size >= 2) zoomRatio.value.coerceIn(points.first(), points.last()) else 1f
                Log.d("Inference", "zoom range $min..$max, stops $points")
            },
            onFrame = { bitmap, started -> onFrame(bitmap, started) },
            onError = { toast(it) },
        )
        sourceRef.set(source)
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> source.start()
                Lifecycle.Event.ON_STOP -> {
                    source.stop(wait = true)
                    // No more frames will arrive to end a recording, so finish it here.
                    stopRecording()
                }
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            source.stop(wait = true)
            sourceRef.set(null)
            runCatching { stopRecording() }
            detector?.close()
        }
    }

    Box(Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize().padding(contentPadding).background(Color.Black)) {
            // The picture and its overlays; pinching here zooms the camera.
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .pointerInput(Unit) {
                        detectTransformGestures { _, _, zoom, _ -> if (zoom != 1f) setZoom(zoomRatio.value * zoom) }
                    },
            ) {
                shown?.let { FrameView(it, Modifier.fillMaxSize()) }
                Row(
                    modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    if (!loading && frameTimes.size > 1) {
                        FrameTimePlot(frameTimes)
                        Spacer(Modifier.width(8.dp))
                    }
                    Text(
                        text = when {
                            loadFailed -> "Model failed to load"
                            loading -> "Loading…"
                            !processing -> "Paused"
                            else -> "${result.inferenceMs} ms"
                        },
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium,
                    )
                }
                Column(
                    modifier = Modifier.align(Alignment.TopEnd).padding(8.dp),
                    horizontalAlignment = Alignment.End,
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    ModelSelector(selectedModel, models, onSelect = selectModel, onManage = { showManage = true }, onAbout = { showAbout = true })
                }
            }
            // Controls below the picture, never on top of it.
            Column(
                modifier = Modifier.fillMaxWidth().height(156.dp).padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.Center,
            ) {
                if (!loading) ThresholdSlider(
                    value = threshold,
                    onChange = { threshold = it; thresholdRef.set(it) },
                    onChangeFinished = { prefs.edit().putFloat("threshold", threshold).apply() },
                )
                if (!loading && zoomPoints.value.size >= 2) {
                    ZoomSelector(zoomPoints.value, zoomRatio.value, onChange = setZoom)
                }
                if (!loading) Row(
                    modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    ProcessingButton(processing, onClick = {
                        processing = !processing
                        processingFlag.set(processing)
                        // Pausing also ends a recording (the analyzer thread saves it on the next frame).
                        if (!processing) {
                            recordFlag.set(false)
                            recording = false
                        }
                    })
                    RecordButton(
                        recording = recording,
                        elapsedMs = recordedMs,
                        enabled = processing,
                        onClick = {
                            recording = !recording
                            if (recording) recordedMs = 0
                            recordFlag.set(recording)
                        },
                    )
                    Spacer(Modifier.weight(1f))
                    GpuBadge(usingGpu)
                }
            }
        }
        if (showManage) {
            ManageModelsDialog(
                models = models,
                store = store,
                installer = installer,
                onChanged = { models = store.all() },
                onDeleted = { deleted -> if (deleted.id == selectedModel) selectModel(DEFAULT_MODEL) },
                onDismiss = { showManage = false },
            )
        }
        if (showAbout) AboutScreen(onDismiss = { showAbout = false })
    }
}

/** The model name with a small arrow; tapping it opens a menu to switch model, and below a divider "Manage models". */
@Composable
private fun ModelSelector(
    selected: String,
    models: List<ModelInfo>,
    onSelect: (String) -> Unit,
    onManage: () -> Unit,
    onAbout: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier) {
        Row(
            modifier = Modifier
                .background(Color(0x99000000), RoundedCornerShape(16.dp))
                .clickable { expanded = true }
                .padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(models.firstOrNull { it.id == selected }?.displayName ?: selected, color = Color.White, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.width(6.dp))
            Text("▾", color = Color.White, style = MaterialTheme.typography.titleMedium)
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            models.forEach { model ->
                DropdownMenuItem(
                    text = { Text(model.displayName + if (model.id == selected) "  ✓" else "") },
                    onClick = {
                        expanded = false
                        if (model.id != selected) onSelect(model.id)
                    },
                )
            }
            HorizontalDivider()
            DropdownMenuItem(
                text = { Text("Manage models…") },
                onClick = {
                    expanded = false
                    onManage()
                },
            )
            DropdownMenuItem(
                text = { Text("About and licences…") },
                onClick = {
                    expanded = false
                    onAbout()
                },
            )
        }
    }
}

/**
 * Minimal zoom control: a thin line with a dot and label per zoom stop (evenly spaced) and a white thumb at the current zoom.
 * Tap to jump to the nearest stop, drag to slide; the thumb also follows pinch-to-zoom.
 */
@Composable
private fun ZoomSelector(points: List<Float>, ratio: Float, onChange: (Float) -> Unit, modifier: Modifier = Modifier) {
    val density = LocalDensity.current
    val labelPaint = remember(density) {
        android.graphics.Paint().apply {
            textSize = with(density) { 12.sp.toPx() }
            textAlign = android.graphics.Paint.Align.CENTER
            isAntiAlias = true
        }
    }
    val line = Color(0xFF9CA3AF)
    val position = ZoomPresets.position(points, ratio)
    val active = (position * (points.size - 1)).roundToInt()

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(44.dp)
            .pointerInput(points) {
                val pad = 44.dp.toPx()
                detectTapGestures { tap ->
                    val fraction = (tap.x - pad) / (size.width - 2 * pad)
                    onChange(ZoomPresets.nearestStop(points, fraction))
                }
            }
            .pointerInput(points) {
                val pad = 44.dp.toPx()
                detectHorizontalDragGestures { change, _ ->
                    val fraction = (change.position.x - pad) / (size.width - 2 * pad)
                    onChange(ZoomPresets.ratioAt(points, fraction))
                }
            },
    ) {
        val pad = 44.dp.toPx()
        val left = pad
        val width = size.width - 2 * pad
        val lineY = size.height - 10.dp.toPx()
        drawLine(line, Offset(left, lineY), Offset(left + width, lineY), strokeWidth = 1.5.dp.toPx())
        points.forEachIndexed { i, stop ->
            val x = left + width * i / (points.size - 1)
            drawCircle(line, radius = 3.dp.toPx(), center = Offset(x, lineY))
            labelPaint.color = if (i == active) android.graphics.Color.WHITE else android.graphics.Color.LTGRAY
            drawContext.canvas.nativeCanvas.drawText(ZoomPresets.label(stop), x, lineY - 14.dp.toPx(), labelPaint)
        }
        drawCircle(Color.White, radius = 7.dp.toPx(), center = Offset(left + width * position, lineY))
    }
}

/**
 * Tiny line plot of the last frame times: gray axes, two labelled ticks on the vertical axis (top = auto-scaled maximum,
 * and half of it), none on the horizontal axis. The line grows from the left until [FRAME_HISTORY] frames are shown.
 */
@Composable
private fun FrameTimePlot(times: List<Long>, modifier: Modifier = Modifier) {
    val density = LocalDensity.current
    val labelPaint = remember(density) {
        android.graphics.Paint().apply {
            color = android.graphics.Color.LTGRAY
            textSize = with(density) { 10.sp.toPx() }
            textAlign = android.graphics.Paint.Align.RIGHT
            isAntiAlias = true
        }
    }
    val axis = Color(0xFF9CA3AF)
    Canvas(
        modifier = modifier
            .size(width = 130.dp, height = 56.dp)
            .background(Color(0x66000000), RoundedCornerShape(6.dp)),
    ) {
        val top = 8.dp.toPx()
        val bottom = size.height - 6.dp.toPx()
        val left = 30.dp.toPx()
        val right = size.width - 6.dp.toPx()
        val max = FrameTimeHistory.niceMax(times.max())
        fun y(ms: Float) = bottom - (bottom - top) * ms / max

        drawLine(axis, Offset(left, top), Offset(left, bottom), strokeWidth = 1.dp.toPx())
        drawLine(axis, Offset(left, bottom), Offset(right, bottom), strokeWidth = 1.dp.toPx())
        val tick = 3.dp.toPx()
        val textOffset = labelPaint.textSize / 3f
        for (value in longArrayOf(max, max / 2)) {
            val ty = y(value.toFloat())
            drawLine(axis, Offset(left - tick, ty), Offset(left, ty), strokeWidth = 1.dp.toPx())
            drawContext.canvas.nativeCanvas.drawText("$value", left - tick - 2.dp.toPx(), ty + textOffset, labelPaint)
        }

        val path = Path()
        times.forEachIndexed { i, ms ->
            val x = left + (right - left) * i / (FRAME_HISTORY - 1)
            val py = y(ms.toFloat())
            if (i == 0) path.moveTo(x, py) else path.lineTo(x, py)
        }
        drawPath(path, Color.White, style = Stroke(width = 1.5.dp.toPx()))
    }
}

/** Round-cornered button that pauses or resumes the detection: shows "Stop" while processing and "Start" while paused. */
@Composable
private fun ProcessingButton(processing: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .background(Color(0x99000000), RoundedCornerShape(24.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            painter = painterResource(if (processing) R.drawable.ic_stop else R.drawable.ic_play),
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size(24.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(if (processing) "Stop" else "Start", color = Color.White, style = MaterialTheme.typography.labelLarge)
    }
}

/** Starts or stops recording the video with the boxes: a red dot and "Record", or a red square and the elapsed time while recording. */
@Composable
private fun RecordButton(recording: Boolean, elapsedMs: Long, enabled: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val red = Color(0xFFF87171)
    Row(
        modifier = modifier
            .alpha(if (enabled) 1f else 0.4f)
            .background(Color(0x99000000), RoundedCornerShape(24.dp))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            painter = painterResource(if (recording) R.drawable.ic_stop else R.drawable.ic_record),
            contentDescription = if (recording) "Stop recording" else "Start recording",
            tint = red,
            modifier = Modifier.size(24.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(if (recording) RecordingFormat.elapsed(elapsedMs) else "Record", color = Color.White, style = MaterialTheme.typography.labelLarge)
    }
}

/** A small chip icon with a label: teal "GPU" when the model runs on the GPU, amber "No GPU" when it fell back. */
@Composable
private fun GpuBadge(usingGpu: Boolean, modifier: Modifier = Modifier) {
    val color = if (usingGpu) Color(0xFF2DD4BF) else Color(0xFFFBBF24)
    Row(
        modifier = modifier
            .background(Color(0x99000000), RoundedCornerShape(16.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            painter = painterResource(R.drawable.ic_chip),
            contentDescription = if (usingGpu) "Running on the GPU" else "Not running on the GPU",
            tint = color,
            modifier = Modifier.size(20.dp),
        )
        Spacer(Modifier.width(6.dp))
        Text(if (usingGpu) "GPU" else "No GPU", color = color, style = MaterialTheme.typography.labelLarge)
    }
}

/** An analysed camera frame (upright) and the detections computed on it. */
private class ShownFrame(val bitmap: Bitmap, val detections: List<Detection>)

/** Draws the frame letterboxed in the available space, with its boxes on top. */
@Composable
private fun FrameView(frame: ShownFrame, modifier: Modifier = Modifier) {
    val renderer = remember { DetectionRenderer() }
    val paint = remember { android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG) }

    Canvas(modifier = modifier) {
        val aspect = frame.bitmap.width.toFloat() / frame.bitmap.height
        val frameH = min(size.width / aspect, size.height)
        val frameW = frameH * aspect
        val originX = (size.width - frameW) / 2f
        val originY = (size.height - frameH) / 2f

        val canvas = drawContext.canvas.nativeCanvas
        canvas.drawBitmap(frame.bitmap, null, android.graphics.RectF(originX, originY, originX + frameW, originY + frameH), paint)
        renderer.draw(canvas, frame.detections, originX, originY, frameW, frameH)
    }
}


/** Minimal confidence control above the zoom line: a short label with the threshold, and a thin line with a small thumb, in 1% steps. Tap or drag. */
@Composable
private fun ThresholdSlider(value: Float, onChange: (Float) -> Unit, onChangeFinished: () -> Unit, modifier: Modifier = Modifier) {
    val line = Color(0xFF9CA3AF)
    Row(modifier = modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text("Conf. ${(value * 100).roundToInt()}%", color = Color.White, style = MaterialTheme.typography.labelMedium, modifier = Modifier.width(76.dp))
        Canvas(
            modifier = Modifier
                .weight(1f)
                .height(28.dp)
                .pointerInput(Unit) {
                    val pad = 8.dp.toPx()
                    fun at(x: Float) = (THRESHOLD_MIN + (THRESHOLD_MAX - THRESHOLD_MIN) * ((x - pad) / (size.width - 2 * pad)).coerceIn(0f, 1f))
                        .let { (it * 100).roundToInt() / 100f }
                    detectTapGestures { tap ->
                        onChange(at(tap.x))
                        onChangeFinished()
                    }
                }
                .pointerInput(Unit) {
                    val pad = 8.dp.toPx()
                    fun at(x: Float) = (THRESHOLD_MIN + (THRESHOLD_MAX - THRESHOLD_MIN) * ((x - pad) / (size.width - 2 * pad)).coerceIn(0f, 1f))
                        .let { (it * 100).roundToInt() / 100f }
                    detectHorizontalDragGestures(onDragEnd = onChangeFinished) { change, _ -> onChange(at(change.position.x)) }
                },
        ) {
            val pad = 8.dp.toPx()
            val width = size.width - 2 * pad
            val x = pad + width * (value - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN)
            drawLine(line, Offset(pad, size.height / 2), Offset(pad + width, size.height / 2), strokeWidth = 1.dp.toPx())
            drawCircle(Color.White, radius = 5.dp.toPx(), center = Offset(x, size.height / 2))
        }
    }
}
