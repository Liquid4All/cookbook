// Minimal JNI wrapper around llama.cpp: load a text model once, then generate
// output that is constrained to a JSON schema.
#include <jni.h>
#include <algorithm>
#include <string>
#include <unistd.h>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static common_chat_templates_ptr g_templates;

constexpr int N_CTX = 4096;
constexpr int N_BATCH = 512;
constexpr int N_PREDICT = 1024;

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
Java_ai_liquid_recipegenerator_LlamaJson_nativeInit(JNIEnv *env, jobject, jstring native_lib_dir) {
    // Load the CPU backend variants that ship as separate .so files
    ggml_backend_load_all_from_path(to_string(env, native_lib_dir).c_str());
    llama_backend_init();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_liquid_recipegenerator_LlamaJson_nativeLoad(JNIEnv *env, jobject, jstring model_path) {
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

    g_templates = common_chat_templates_init(g_model, "");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_recipegenerator_LlamaJson_nativeGenerate(JNIEnv *env, jobject, jstring system_prompt,
                                                        jstring user_prompt, jstring json_schema,
                                                        jobject callback) {
    // Every request starts from an empty context
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Chat template: system message, then the user message
    common_chat_msg system_msg;
    system_msg.role = "system";
    system_msg.content = to_string(env, system_prompt);
    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = to_string(env, user_prompt);
    std::string prompt = common_chat_format_single(g_templates.get(), {}, system_msg, false, false);
    prompt += common_chat_format_single(g_templates.get(), {system_msg}, user_msg, true, false);

    // Process the prompt in batches
    llama_tokens tokens = common_tokenize(g_ctx, prompt, true, true);
    if ((int) tokens.size() >= N_CTX) return 1;
    common_batch batch(g_ctx);
    for (size_t i = 0; i < tokens.size(); i += N_BATCH) {
        batch.clear();
        const size_t end = std::min(tokens.size(), i + N_BATCH);
        for (size_t j = i; j < end; ++j) batch.add(tokens[j], (llama_pos) j, 0, j == tokens.size() - 1);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) return 2;
    }
    llama_pos n_past = (llama_pos) tokens.size();

    // Sampling parameters from the LFM2.5-1.2B-Instruct model card, plus a
    // grammar that only allows output matching the JSON schema
    common_params_sampling sparams;
    sparams.temp = 0.1f;
    sparams.top_k = 50;
    sparams.top_p = 1.0f;
    sparams.min_p = 0.0f;
    sparams.penalty_repeat = 1.05f;
    sparams.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
                       json_schema_to_grammar(common_json::parse(to_string(env, json_schema)))};
    common_sampler *sampler = common_sampler_init(g_model, sparams);
    if (!sampler) return 3;

    // Generate and stream each complete UTF-8 piece to the Kotlin callback
    jmethodID on_token = env->GetMethodID(env->GetObjectClass(callback), "onToken", "([B)V");
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    std::string pending;
    int rc = 0;
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
            rc = 4;
            break;
        }
    }
    common_sampler_free(sampler);
    return rc;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_recipegenerator_LlamaJson_nativeUnload(JNIEnv *, jobject) {
    g_templates.reset();
    if (g_ctx) llama_free(g_ctx);
    if (g_model) llama_model_free(g_model);
    g_ctx = nullptr;
    g_model = nullptr;
}
