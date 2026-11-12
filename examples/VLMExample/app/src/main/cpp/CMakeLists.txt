cmake_minimum_required(VERSION 3.22.1)

project("vlm-jni" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Same CPU settings as llama.cpp's examples/llama.android/lib
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_SYSTEM_ARCH "ARM")
    set(GGML_CPU_KLEIDIAI ON)
    set(GGML_OPENMP ON)
elseif(ANDROID_ABI STREQUAL "x86_64")
    set(GGML_SYSTEM_ARCH "x86")
    set(GGML_CPU_KLEIDIAI OFF)
    set(GGML_OPENMP OFF)
endif()

# llama.cpp submodule, plus its mtmd library for image input
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../llama.cpp)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_SRC} build-llama)

add_library(vlm-jni SHARED vlm_jni.cpp)

target_include_directories(vlm-jni PRIVATE
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/tools/mtmd)

target_link_libraries(vlm-jni
        llama
        llama-common
        mtmd
        android
        log)
