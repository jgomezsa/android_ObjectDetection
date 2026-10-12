package es.jrg.objectdetection

import android.content.Context
import android.content.Intent
import android.net.Uri
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
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

private fun openLink(context: Context, url: String) {
    try {
        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    } catch (e: Exception) {
        // No browser installed: nothing to do.
    }
}

/** App version, the not-open-source statement, links to the privacy policy and the source, and the third-party licence notices. An overlay like [ManageModelsDialog]. */
@Composable
fun AboutScreen(onDismiss: () -> Unit) {
    val context = LocalContext.current
    val version = remember(context) {
        try {
            context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: ""
        } catch (e: Exception) {
            ""
        }
    }
    BackHandler(onBack = onDismiss)
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
                Text("About", style = MaterialTheme.typography.headlineSmall)
                if (version.isNotEmpty()) Text("Version $version", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                Spacer(Modifier.height(12.dp))
                Column(
                    modifier = Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(
                        "Detects objects in the camera image on your phone. The camera image and the detections never leave the device; " +
                            "the internet is used only when you download a model you chose.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Text(LICENSE_STATEMENT, style = MaterialTheme.typography.bodyMedium)
                    Row {
                        TextButton(onClick = { openLink(context, PRIVACY_POLICY_URL) }) { Text("Privacy policy") }
                        TextButton(onClick = { openLink(context, SOURCE_URL) }) { Text("Source code") }
                    }
                    HorizontalDivider(Modifier.padding(vertical = 4.dp))
                    Text("Third-party software and models", style = MaterialTheme.typography.titleSmall)
                    Notices.all.forEach { notice ->
                        Column(Modifier.fillMaxWidth()) {
                            Text(notice.name, style = MaterialTheme.typography.bodyMedium)
                            Text(notice.licence, style = MaterialTheme.typography.bodySmall, color = Color.DarkGray)
                            Text(
                                notice.url,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.pointerInput(notice.url) { detectTapGestures { openLink(context, notice.url) } },
                            )
                        }
                    }
                    Text(
                        "Models you download yourself have their own licences; check them before using or sharing them.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.DarkGray,
                    )
                }
                Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.End) {
                    TextButton(onClick = onDismiss) { Text("Close") }
                }
            }
        }
    }
}
