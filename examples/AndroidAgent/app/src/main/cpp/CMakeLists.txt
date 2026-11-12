cmake_minimum_required(VERSION 3.22.1)

project("llama-agent-jni" LANGUAGES C CXX)

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

# llama.cpp submodule
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../llama.cpp)
add_subdirectory(${LLAMA_SRC} build-llama)

add_library(llama-agent-jni SHARED llama_agent_jni.cpp)

target_include_directories(llama-agent-jni PRIVATE
        ${LLAMA_SRC}/common)

target_link_libraries(llama-agent-jni
        llama
        llama-common
        android
        log)
