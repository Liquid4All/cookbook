// Minimal JNI wrapper around llama.cpp and its mtmd library:
// load a vision model plus its projector once, then describe images.
#include "jni_bytes.h"
#include <mutex>
#include <memory>
#include <algorithm>
#include <string>
#include <unistd.h>
#include <vector>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

struct State {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    mtmd_context *mtmd = nullptr;
    common_chat_templates_ptr templates;
    ~State() { reset(); }
    void reset() {
        templates.reset();
        if (mtmd) mtmd_free(mtmd);
        mtmd = nullptr;
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        ctx = nullptr;
        model = nullptr;
    }
};

constexpr int N_CTX = 4096;
constexpr int N_BATCH = 512;

// Length of the longest prefix of `s` that ends on a complete UTF-8 character
static size_t complete_utf8_prefix(const std::string &s) {
    size_t i = s.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = s[i - 1];
        if ((c & 0xC0) != 0x80) {
            size_t need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
            return back + 1 >= need ? s.size() : i - 1;
        }
        --i;
        ++back;
    }
    return s.size();
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_liquid_android_runtime_NativeVisionModel_nativeInit(JNIEnv *env, jobject, jbyteArray native_lib_dir) try {
    static std::once_flag initialized;
    std::call_once(initialized, [&] {
        configure_logging();
        ggml_backend_load_all_from_path(from_bytes(env, native_lib_dir).c_str());
        llama_backend_init();
    });
    return reinterpret_cast<jlong>(new State());
}
catch (const std::exception &error) { throw_native_error(env, error.what()); return 0; }

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_liquid_android_runtime_NativeVisionModel_nativeLoad(JNIEnv *env, jobject, jlong handle, jbyteArray model_path, jbyteArray mmproj_path) try {
    auto &s = *reinterpret_cast<State *>(handle);
    s.model = llama_model_load_from_file(from_bytes(env, model_path).c_str(), llama_model_default_params());
    if (!s.model) return JNI_FALSE;

    const int n_threads = std::max(2, std::min(4, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_CTX;
    cparams.n_batch = N_BATCH;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    s.ctx = llama_init_from_model(s.model, cparams);
    if (!s.ctx) { s.reset(); return JNI_FALSE; }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false;
    mparams.n_threads = n_threads;
    s.mtmd = mtmd_init_from_file(from_bytes(env, mmproj_path).c_str(), s.model, mparams);
    if (!s.mtmd) { s.reset(); return JNI_FALSE; }

    s.templates = common_chat_templates_init(s.model, "");
    return JNI_TRUE;
}
catch (const std::exception &error) { reinterpret_cast<State *>(handle)->reset(); throw_native_error(env, error.what()); return JNI_FALSE; }

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_android_runtime_NativeVisionModel_nativeGenerate(JNIEnv *env, jobject, jlong handle, jbyteArray system_prompt,
                                                  jbyteArray user_prompt, jbyteArray image, jobject callback, jint max_tokens) try {
    auto &s = *reinterpret_cast<State *>(handle);
    if (max_tokens <= 0 || max_tokens > N_CTX) return 7;
    auto should_continue = env->GetMethodID(env->GetObjectClass(callback), "shouldContinue", "()Z");
    // Every request starts from an empty context
    llama_memory_clear(llama_get_memory(s.ctx), true);

    // Chat template: system message, then the user message with the image marker
    common_chat_msg system_msg;
    system_msg.role = "system";
    system_msg.content = from_bytes(env, system_prompt);
    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = std::string(mtmd_default_marker()) + "\n" + from_bytes(env, user_prompt);
    common_chat_templates_inputs inputs;
    inputs.messages = {system_msg, user_msg};
    inputs.add_generation_prompt = true;
    inputs.add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(s.model));
    std::string prompt = common_chat_templates_apply(s.templates.get(), inputs).prompt.str();

    // Decode the JPEG bytes into a bitmap
    std::vector<unsigned char> bytes(env->GetArrayLength(image));
    env->GetByteArrayRegion(image, 0, (jsize) bytes.size(), reinterpret_cast<jbyte *>(bytes.data()));
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
            s.mtmd, bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default());
    if (!wrapper.bitmap) return 10;

    // Tokenize text and image together, then run them through the model
    mtmd_input_text text = {prompt.c_str(), prompt.size(), true, true};
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const mtmd_bitmap *bitmaps[] = {wrapper.bitmap};
    llama_pos n_past = 0;
    int rc = mtmd_tokenize(s.mtmd, chunks, &text, bitmaps, 1);
    if (rc == 0) rc = mtmd_helper_eval_chunks(s.mtmd, s.ctx, chunks, 0, 0, N_BATCH, true, &n_past);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(wrapper.bitmap);
    if (rc != 0) return 2;

    // Sampling parameters from the LFM2.5-VL-1.6B model card
    common_params_sampling sparams;
    sparams.temp = 0.1f;
    sparams.min_p = 0.15f;
    sparams.penalty_repeat = 1.05f;
    sparams.top_k = 0;
    sparams.top_p = 1.0f;
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler_owner(
        common_sampler_init(s.model, sparams), common_sampler_free);
    auto *sampler = sampler_owner.get();
    if (!sampler) return 3;

    // Generate and stream each complete UTF-8 piece to the Kotlin callback
    jmethodID on_token = env->GetMethodID(env->GetObjectClass(callback), "onToken", "([B)Z");
    const llama_vocab *vocab = llama_model_get_vocab(s.model);
    common_batch batch(s.ctx);
    std::string pending;
    bool completed = false;
    for (int i = 0; i < max_tokens && n_past < N_CTX; ++i) {
        if (!env->CallBooleanMethod(callback, should_continue)) { rc = 5; break; }
        llama_token token = common_sampler_sample(sampler, s.ctx, -1);
        common_sampler_accept(sampler, token, true);
        if (llama_vocab_is_eog(vocab, token)) { completed = true; break; }

        pending += common_token_to_piece(s.ctx, token, false);
        size_t n = complete_utf8_prefix(pending);
        if (n > 0) {
            jbyteArray piece = env->NewByteArray((jsize) n);
            env->SetByteArrayRegion(piece, 0, (jsize) n, reinterpret_cast<const jbyte *>(pending.data()));
            const bool keep_going = env->CallBooleanMethod(callback, on_token, piece);
            env->DeleteLocalRef(piece);
            if (!keep_going || env->ExceptionCheck()) { rc = 5; break; }
            pending.erase(0, n);
        }

        batch.clear();
        batch.add(token, n_past++, 0, true);
        if (llama_process(s.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            rc = 3;
            break;
        }
    }
    return rc != 0 ? rc : completed ? 0 : 6;
}
catch (const std::exception &error) { throw_native_error(env, error.what()); return 8; }

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_android_runtime_NativeVisionModel_nativeUnload(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<State *>(handle);
}
