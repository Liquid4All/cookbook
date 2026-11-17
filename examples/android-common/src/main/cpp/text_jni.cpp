// Minimal JNI wrapper around llama.cpp: load a text model once, then generate
// output that is constrained to a JSON schema.
#include "jni_bytes.h"
#include <mutex>
#include <memory>
#include <algorithm>
#include <string>
#include <unistd.h>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"

struct State {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    common_chat_templates_ptr templates;
    ~State() { reset(); }
    void reset() {
        templates.reset();
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        ctx = nullptr;
        model = nullptr;
    }
};

constexpr int N_CTX = 8192;
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
Java_ai_liquid_android_runtime_NativeTextModel_nativeInit(JNIEnv *env, jobject, jbyteArray native_lib_dir) try {
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
Java_ai_liquid_android_runtime_NativeTextModel_nativeLoad(JNIEnv *env, jobject, jlong handle, jbyteArray model_path) try {
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

    s.templates = common_chat_templates_init(s.model, "");
    return JNI_TRUE;
}
catch (const std::exception &error) { reinterpret_cast<State *>(handle)->reset(); throw_native_error(env, error.what()); return JNI_FALSE; }

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_android_runtime_NativeTextModel_nativeGenerate(JNIEnv *env, jobject, jlong handle, jbyteArray system_prompt,
                                                        jbyteArray user_prompt, jbyteArray json_schema,
                                                        jobject callback, jint max_tokens) try {
    auto &s = *reinterpret_cast<State *>(handle);
    if (max_tokens <= 0 || max_tokens > N_CTX) return 7;
    auto should_continue = env->GetMethodID(env->GetObjectClass(callback), "shouldContinue", "()Z");
    // Every request starts from an empty context
    llama_memory_clear(llama_get_memory(s.ctx), true);

    // Chat template: system message, then the user message
    common_chat_msg system_msg;
    system_msg.role = "system";
    system_msg.content = from_bytes(env, system_prompt);
    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = from_bytes(env, user_prompt);
    common_chat_templates_inputs inputs;
    inputs.messages = {system_msg, user_msg};
    inputs.add_generation_prompt = true;
    inputs.add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(s.model));
    std::string prompt = common_chat_templates_apply(s.templates.get(), inputs).prompt.str();

    // Process the prompt in batches
    llama_tokens tokens = common_tokenize(s.ctx, prompt, true, true);
    if ((int) tokens.size() >= N_CTX) return 1;
    common_batch batch(s.ctx);
    for (size_t i = 0; i < tokens.size(); i += N_BATCH) {
        if (!env->CallBooleanMethod(callback, should_continue)) return 5;
        batch.clear();
        const size_t end = std::min(tokens.size(), i + N_BATCH);
        for (size_t j = i; j < end; ++j) batch.add(tokens[j], (llama_pos) j, 0, j == tokens.size() - 1);
        if (llama_process(s.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) return 2;
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
    if (!from_bytes(env, json_schema).empty()) sparams.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
                       json_schema_to_grammar(common_json::parse(from_bytes(env, json_schema)))};
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler_owner(
        common_sampler_init(s.model, sparams), common_sampler_free);
    auto *sampler = sampler_owner.get();
    if (!sampler) return 3;

    // Generate and stream each complete UTF-8 piece to the Kotlin callback
    jmethodID on_token = env->GetMethodID(env->GetObjectClass(callback), "onToken", "([B)Z");
    const llama_vocab *vocab = llama_model_get_vocab(s.model);
    std::string pending;
    int rc = 0;
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
            rc = 4;
            break;
        }
    }
    return rc != 0 ? rc : completed ? 0 : 6;
}
catch (const std::exception &error) { throw_native_error(env, error.what()); return 8; }

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_android_runtime_NativeTextModel_nativeUnload(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<State *>(handle);
}
