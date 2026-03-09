pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // LeapSDK repository
        maven {
            url = uri("https://maven.pkg.github.com/liquid4all/LeapSDK-Android")
            credentials {
                username = System.getenv("GITHUB_USERNAME") ?: providers.gradleProperty("gpr.user").orNull ?: ""
                password = System.getenv("GITHUB_TOKEN") ?: providers.gradleProperty("gpr.token").orNull ?: ""
            }
        }
    }
}

rootProject.name = "BrowserControlDemo"
include(":app")
