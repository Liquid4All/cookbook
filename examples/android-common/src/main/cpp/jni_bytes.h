#pragma once
#include <jni.h>
#include <string>
#include "llama.h"
#ifdef __ANDROID__
#include <android/log.h>
#endif

inline void configure_logging() {
#ifdef __ANDROID__
    llama_log_set([](ggml_log_level level, const char *text, void *) {
        const int priority = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
                             level == GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
        __android_log_print(priority, "LiquidLlama", "%s", text);
    }, nullptr);
#endif
}

// Kotlin supplies standard UTF-8 bytes; JNI's Modified UTF-8 is never used for model text.
inline std::string from_bytes(JNIEnv *env, jbyteArray value) {
    const auto size = env->GetArrayLength(value);
    std::string result(size, '\0');
    env->GetByteArrayRegion(value, 0, size, reinterpret_cast<jbyte *>(result.data()));
    return result;
}

inline jbyteArray to_bytes(JNIEnv *env, const std::string &value) {
    auto result = env->NewByteArray(static_cast<jsize>(value.size()));
    if (result) env->SetByteArrayRegion(result, 0, static_cast<jsize>(value.size()),
                                      reinterpret_cast<const jbyte *>(value.data()));
    return result;
}

inline void throw_native_error(JNIEnv *env, const std::string &message) {
    if (env->ExceptionCheck()) return;
    auto bytes = to_bytes(env, message);
    auto charset = env->NewStringUTF("UTF-8");
    auto string_class = env->FindClass("java/lang/String");
    auto constructor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    auto text = env->NewObject(string_class, constructor, bytes, charset);
    auto error_class = env->FindClass("java/lang/IllegalStateException");
    auto error_constructor = env->GetMethodID(error_class, "<init>", "(Ljava/lang/String;)V");
    auto error = static_cast<jthrowable>(env->NewObject(error_class, error_constructor, text));
    env->Throw(error);
}
