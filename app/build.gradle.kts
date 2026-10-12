import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "es.jrg.objectdetection"
    compileSdk = 35

    defaultConfig {
        applicationId = "es.jrg.objectdetection"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }

    // Release signing: the upload key lives in keystore.properties (git-ignored; see TECHNICAL_DOCUMENTATION.md, Publishing).
    val keystoreProperties = Properties().apply {
        val file = rootProject.file("keystore.properties")
        if (file.isFile) file.inputStream().use { load(it) }
    }
    signingConfigs {
        if (!keystoreProperties.isEmpty) {
            create("release") {
                storeFile = rootProject.file(keystoreProperties.getProperty("storeFile"))
                storePassword = keystoreProperties.getProperty("storePassword")
                keyAlias = keystoreProperties.getProperty("keyAlias")
                keyPassword = keystoreProperties.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Phones sold for years are 64-bit ARM; leaving out the other native libraries makes the app much smaller.
            ndk { abiFilters += "arm64-v8a" }
            signingConfigs.findByName("release")?.let { signingConfig = it }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        // LiteRT 2.3.0 is built with a newer Kotlin than 2.0.21; its public API is plain enough to read anyway.
        freeCompilerArgs += "-Xskip-metadata-version-check"
    }
    lint {
        // The lint bundled with this AGP crashes on the newer Kotlin metadata that LiteRT is built with (see -Xskip-metadata-version-check).
        checkReleaseBuilds = false
    }
    buildFeatures {
        compose = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-gpu:2.3.0")

    testImplementation("junit:junit:4.13.2")
}
