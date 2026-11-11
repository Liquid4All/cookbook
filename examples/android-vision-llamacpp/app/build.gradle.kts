plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "ai.liquid.visiondemo"
    compileSdk = 36
    ndkVersion = "28.2.13676358"
    defaultConfig {
        applicationId = "ai.liquid.visiondemo"
        minSdk = 33
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild { cmake {
            arguments += listOf("-DBUILD_SHARED_LIBS=ON", "-DLLAMA_BUILD_COMMON=ON",
                "-DLLAMA_BUILD_MTMD=ON", "-DMTMD_VIDEO=OFF", "-DLLAMA_OPENSSL=OFF",
                "-DGGML_NATIVE=OFF", "-DGGML_BACKEND_DL=ON",
                "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_LLAMAFILE=OFF")
        } }
    }
    buildFeatures { compose = true }
    externalNativeBuild { cmake { path("src/main/cpp/CMakeLists.txt"); version = "3.31.6" } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlin { jvmToolchain(17) }
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.material3)
}
