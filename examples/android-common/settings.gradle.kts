pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
    plugins {
        id("com.android.library") version "8.13.2"
        id("org.jetbrains.kotlin.android") version "2.3.20"
    }
}
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "LiquidAndroidRuntime"
