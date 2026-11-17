plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// Independent example projects must not overwrite each other's AARs or native outputs.
if (rootProject != project) {
    layout.buildDirectory.set(rootProject.layout.projectDirectory.dir("build/runtime"))
}

val nativeTarget = when (rootProject.name) {
    "LiquidAndroidRuntime" -> "all"
    "AndroidAgent", "LeapKoogAgent" -> "agent"
    "VLMExample", "VLMTestApp", "VLM Test App" -> "vision"
    else -> "text"
}
val llamaSource = providers.gradleProperty("llamaSource").orNull
    ?: rootProject.file("llama.cpp").takeIf { it.resolve("CMakeLists.txt").exists() }?.absolutePath
    ?: project.file("../SloganApp/llama.cpp").absolutePath

android {
    namespace = "ai.liquid.android.runtime"
    compileSdk = 36
    ndkVersion = "29.0.13113456"
    defaultConfig {
        minSdk = 31
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild.cmake {
            arguments += listOf("-DLLAMA_SRC=$llamaSource", "-DLIQUID_RUNTIME=$nativeTarget", "-DCMAKE_BUILD_TYPE=Release")
        }
    }
    externalNativeBuild.cmake {
        path("src/main/cpp/CMakeLists.txt")
        version = "3.31.6"
        buildStagingDirectory = rootProject.file(".cxx/runtime")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin { jvmToolchain(17) }
    testOptions { unitTests.isReturnDefaultValues = true }
    packaging.jniLibs.useLegacyPackaging = true
}

dependencies {
    api("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    api("androidx.lifecycle:lifecycle-viewmodel-ktx:2.10.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
}
