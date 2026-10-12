package es.jrg.objectdetection

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Developer overrides, e.g. `am start ... --es model yolox_tiny --es backend CPU`.
        val model = intent.getStringExtra("model")
        val backend = intent.getStringExtra("backend")?.let { name -> Backend.entries.firstOrNull { it.name.equals(name, ignoreCase = true) } }
        setContent {
            MaterialTheme {
                // Scaffold exposes the system bar insets as innerPadding; apply it to content.
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    CameraPermissionGate(innerPadding, model, backend)
                }
            }
        }
    }
}

@Composable
fun CameraPermissionGate(
    contentPadding: PaddingValues = PaddingValues(),
    model: String? = null,
    backend: Backend? = null,
) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) launcher.launch(Manifest.permission.CAMERA) }

    if (granted) {
        CameraScreen(contentPadding, initialModel = model, forcedBackend = backend)
    } else {
        Column(
            modifier = Modifier.fillMaxSize().padding(contentPadding).padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Camera permission is needed to detect objects.")
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("Grant permission") }
        }
    }
}
