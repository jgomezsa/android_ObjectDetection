package com.example.objectdetection

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import java.util.concurrent.atomic.AtomicBoolean

/** A short explanation of the model files the app accepts, shown in the "Manage models" dialog. */
val MODEL_FORMAT_NOTES = listOf(
    ".tflite runs on the GPU (fast) or the CPU. It needs a fixed-size float model; int8 (quantised) files usually fall back to the CPU or fail to load. An ONNX model has to be converted on a PC first (onnx2tf).",
    ".onnx runs on the CPU only (ONNX Runtime), so it is several times slower than a .tflite on the GPU, but it needs no conversion.",
    "The app recognises the type of model by running it once. Supported: YOLOX, YOLOv8 and YOLO11 (output [1, 4 + classes, 8400]), NMS-free YOLO such as YOLOv10 (output [1, 300, 6]), and DETR-style detectors such as RT-DETR and D-FINE (class logits [1, 300, classes] plus boxes [1, 300, 4], or [1, 300, 4 + classes]). Each type gets its own colour order, value range and resizing.",
    "Not supported: models that need extra inputs, ImageNet-normalised inputs (original DETR, RF-DETR), non-square inputs, SSD, EfficientDet, segmentation, pose or classification models, and archives (.zip, .tar.gz).",
    "Class names: 80 classes use the COCO names. For any other number, give a class names address (a .txt file with one name per line), otherwise boxes say \"class 0\", \"class 1\"...",
    "Download only models you trust and check their licence; some are not free for commercial use.",
)

private sealed interface AddStatus {
    data object Idle : AddStatus
    data class Working(val message: String, val fraction: Float?) : AddStatus
    data class Failed(val message: String) : AddStatus
    data class Added(val name: String) : AddStatus
}

/**
 * Lists the models with a Delete button for the downloaded ones, lets the user add a model from a web address, and explains
 * the model formats. [onChanged] is called after a model was added or deleted so the caller can reload its list.
 * It is drawn as an overlay over the whole screen, inside the safe drawing area (above the keyboard and the system bars), so its buttons
 * are never covered; call it from a `Box` that fills the screen.
 */
@Composable
fun ManageModelsDialog(
    models: List<ModelInfo>,
    store: ModelStore,
    installer: ModelInstaller,
    onChanged: () -> Unit,
    onDeleted: (ModelInfo) -> Unit,
    onDismiss: () -> Unit,
) {
    var modelUrl by remember { mutableStateOf("") }
    var labelsUrl by remember { mutableStateOf("") }
    var status by remember { mutableStateOf<AddStatus>(AddStatus.Idle) }
    var pendingDelete by remember { mutableStateOf<ModelInfo?>(null) }
    val cancelled = remember { AtomicBoolean(false) }
    val working = status is AddStatus.Working

    fun startAdd() {
        cancelled.set(false)
        status = AddStatus.Working("Starting…", null)
        val url = modelUrl
        val labels = labelsUrl
        Thread({
            status = try {
                val added = installer.install(url, labels, { message, fraction -> status = AddStatus.Working(message, fraction) }, { cancelled.get() })
                onChanged()
                AddStatus.Added(added.displayName)
            } catch (e: ModelRejectedException) {
                AddStatus.Failed(e.message ?: "The model was rejected.")
            } catch (e: Throwable) {
                AddStatus.Failed("Could not add the model: ${e.message ?: e.javaClass.simpleName}")
            }
        }, "model-install").start()
    }

    // The back gesture closes the panel (or, first, the keyboard).
    BackHandler(onBack = onDismiss)

    // An overlay in the main window rather than a platform dialog: the keyboard inset is reliable here, so the buttons stay visible.
    run {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0x99000000))
                .pointerInput(Unit) { detectTapGestures { onDismiss() } }
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(16.dp),
            contentAlignment = Alignment.Center,
        ) {
            Surface(
                modifier = Modifier.pointerInput(Unit) { detectTapGestures { } },
                shape = RoundedCornerShape(28.dp),
                tonalElevation = 6.dp,
                color = MaterialTheme.colorScheme.surfaceContainerHigh,
            ) {
                Column(Modifier.padding(24.dp)) {
                    Text("Manage models", style = MaterialTheme.typography.headlineSmall)
                    Spacer(Modifier.height(12.dp))
                    Column(
                        modifier = Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        models.forEach { model ->
                            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                                Column(Modifier.weight(1f)) {
                                    Text(model.displayName, style = MaterialTheme.typography.titleSmall)
                                    val size = if (model.sizeBytes >= 1_000_000) "%.1f MB".format(model.sizeBytes / 1_000_000.0) else "${model.sizeBytes / 1000} KB"
                                    val classes = "${model.numClasses} ${if (model.numClasses == 1) "class" else "classes"}"
                                    Text(
                                        "${model.format.label} • ${model.spec?.family ?: "?"} • $classes • $size" + if (model.builtIn) " • built in" else "",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = Color.Gray,
                                    )
                                }
                                if (!model.builtIn) TextButton(onClick = { pendingDelete = model }) { Text("Delete") }
                            }
                        }
                        HorizontalDivider(Modifier.padding(vertical = 4.dp))
                        Text("Add a model from a web address", style = MaterialTheme.typography.titleSmall)
                        OutlinedTextField(
                            value = modelUrl,
                            onValueChange = { modelUrl = it },
                            label = { Text("Model address (https)") },
                            singleLine = true,
                            enabled = !working,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                            modifier = Modifier.fillMaxWidth(),
                        )
                        OutlinedTextField(
                            value = labelsUrl,
                            onValueChange = { labelsUrl = it },
                            label = { Text("Class names (optional)") },
                            singleLine = true,
                            enabled = !working,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                            modifier = Modifier.fillMaxWidth(),
                        )
                        HorizontalDivider(Modifier.padding(vertical = 4.dp))
                        Text("About model formats", style = MaterialTheme.typography.titleSmall)
                        MODEL_FORMAT_NOTES.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall, color = Color.DarkGray) }
                        Spacer(Modifier.height(4.dp))
                    }
                    // The progress or result of adding a model, and the buttons, are outside the scrolling part so they are always visible.
                    when (val s = status) {
                        is AddStatus.Working -> {
                            Spacer(Modifier.height(8.dp))
                            Text(s.message, style = MaterialTheme.typography.bodySmall)
                            if (s.fraction != null) LinearProgressIndicator(progress = { s.fraction }, modifier = Modifier.fillMaxWidth())
                            else LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        }
                        is AddStatus.Failed -> Text(s.message, color = Color(0xFFB3261E), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(top = 8.dp))
                        is AddStatus.Added -> Text("Added \"${s.name}\". Pick it from the model menu.", color = Color(0xFF1B7F3B), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(top = 8.dp))
                        AddStatus.Idle -> Unit
                    }
                    Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.End) {
                        if (working) {
                            TextButton(onClick = { cancelled.set(true) }) { Text("Cancel download") }
                        } else {
                            TextButton(onClick = { startAdd() }, enabled = modelUrl.isNotBlank()) { Text("Download and add") }
                        }
                        TextButton(onClick = onDismiss) { Text("Close") }
                    }
                }
            }
        }
    }

    pendingDelete?.let { model ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("Delete ${model.displayName}?") },
            text = { Text("The downloaded file is removed from the phone. You can add it again from its address.") },
            confirmButton = {
                TextButton(onClick = {
                    pendingDelete = null
                    if (store.delete(model.id)) {
                        onDeleted(model)
                        onChanged()
                    }
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { pendingDelete = null }) { Text("Cancel") } },
        )
    }
}
