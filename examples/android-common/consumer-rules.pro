-keepclasseswithmembers,includedescriptorclasses class ai.liquid.android.runtime.** {
    native <methods>;
}
-keep class ai.liquid.android.runtime.NativeTextModel$TokenCallback { *; }
-keep class ai.liquid.android.runtime.NativeAgentModel$CancellationCallback { *; }
-keepclassmembers class * implements ai.liquid.android.runtime.NativeTextModel$TokenCallback { *; }
-keepclassmembers class * implements ai.liquid.android.runtime.NativeAgentModel$CancellationCallback { *; }
