// Minimal JNI wrapper around llama.cpp and its mtmd library:
// load a vision model plus its projector once, then describe images.
#include <jni.h>
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

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static mtmd_context *g_mtmd = nullptr;
static common_chat_templates_ptr g_templates;

constexpr int N_CTX = 4096;
constexpr int N_BATCH = 512;
constexpr int N_PREDICT = 512;

static std::string to_string(JNIEnv *env, jstring s) {
    const char *chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

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

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_vlmtestapp_LlamaVlm_nativeInit(JNIEnv *env, jobject, jstring native_lib_dir) {
    // Load the CPU backend variants that ship as separate .so files
    ggml_backend_load_all_from_path(to_string(env, native_lib_dir).c_str());
    llama_backend_init();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_liquid_vlmtestapp_LlamaVlm_nativeLoad(JNIEnv *env, jobject, jstring model_path, jstring mmproj_path) {
    g_model = llama_model_load_from_file(to_string(env, model_path).c_str(), llama_model_default_params());
    if (!g_model) return JNI_FALSE;

    const int n_threads = std::max(2, std::min(4, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_CTX;
    cparams.n_batch = N_BATCH;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) return JNI_FALSE;

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false;
    mparams.n_threads = n_threads;
    g_mtmd = mtmd_init_from_file(to_string(env, mmproj_path).c_str(), g_model, mparams);
    if (!g_mtmd) return JNI_FALSE;

    g_templates = common_chat_templates_init(g_model, "");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_vlmtestapp_LlamaVlm_nativeGenerate(JNIEnv *env, jobject, jstring system_prompt,
                                                  jstring user_prompt, jbyteArray image, jobject callback) {
    // Every request starts from an empty context
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Chat template: system message, then the user message with the image marker
    common_chat_msg system_msg;
    system_msg.role = "system";
    system_msg.content = to_string(env, system_prompt);
    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = std::string(mtmd_default_marker()) + "\n" + to_string(env, user_prompt);
    std::string prompt = common_chat_format_single(g_templates.get(), {}, system_msg, false, false);
    prompt += common_chat_format_single(g_templates.get(), {system_msg}, user_msg, true, false);

    // Decode the JPEG bytes into a bitmap
    std::vector<unsigned char> bytes(env->GetArrayLength(image));
    env->GetByteArrayRegion(image, 0, (jsize) bytes.size(), reinterpret_cast<jbyte *>(bytes.data()));
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
            g_mtmd, bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default());
    if (!wrapper.bitmap) return 1;

    // Tokenize text and image together, then run them through the model
    mtmd_input_text text = {prompt.c_str(), prompt.size(), true, true};
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const mtmd_bitmap *bitmaps[] = {wrapper.bitmap};
    llama_pos n_past = 0;
    int rc = mtmd_tokenize(g_mtmd, chunks, &text, bitmaps, 1);
    if (rc == 0) rc = mtmd_helper_eval_chunks(g_mtmd, g_ctx, chunks, 0, 0, N_BATCH, true, &n_past);
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
    common_sampler *sampler = common_sampler_init(g_model, sparams);

    // Generate and stream each complete UTF-8 piece to the Kotlin callback
    jmethodID on_token = env->GetMethodID(env->GetObjectClass(callback), "onToken", "([B)V");
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    common_batch batch(g_ctx);
    std::string pending;
    for (int i = 0; i < N_PREDICT && n_past < N_CTX; ++i) {
        llama_token token = common_sampler_sample(sampler, g_ctx, -1);
        common_sampler_accept(sampler, token, true);
        if (llama_vocab_is_eog(vocab, token)) break;

        pending += common_token_to_piece(g_ctx, token, false);
        size_t n = complete_utf8_prefix(pending);
        if (n > 0) {
            jbyteArray piece = env->NewByteArray((jsize) n);
            env->SetByteArrayRegion(piece, 0, (jsize) n, reinterpret_cast<const jbyte *>(pending.data()));
            env->CallVoidMethod(callback, on_token, piece);
            env->DeleteLocalRef(piece);
            pending.erase(0, n);
        }

        batch.clear();
        batch.add(token, n_past++, 0, true);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            rc = 3;
            break;
        }
    }
    common_sampler_free(sampler);
    return rc;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_vlmtestapp_LlamaVlm_nativeUnload(JNIEnv *, jobject) {
    g_templates.reset();
    if (g_mtmd) mtmd_free(g_mtmd);
    if (g_ctx) llama_free(g_ctx);
    if (g_model) llama_model_free(g_model);
    g_mtmd = nullptr;
    g_ctx = nullptr;
    g_model = nullptr;
}
