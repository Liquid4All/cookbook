// Minimal JNI wrapper around llama.cpp for tool calling: load a text model once,
// then turn a chat (OpenAI-format messages and tools) into the model's next
// assistant message, with any tool calls already parsed.
#include "jni_bytes.h"
#include <mutex>
#include <memory>
#include <atomic>
#include <algorithm>
#include <string>
#include <unistd.h>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
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

static jbyteArray error_json(JNIEnv *env, const std::string &message) {
    return to_bytes(env, common_json{{"error", message}}.dump());
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_liquid_android_runtime_NativeAgentModel_nativeInit(JNIEnv *env, jobject, jbyteArray native_lib_dir) try {
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
Java_ai_liquid_android_runtime_NativeAgentModel_nativeLoad(JNIEnv *env, jobject, jlong handle, jbyteArray model_path) try {
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

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_liquid_android_runtime_NativeAgentModel_nativeChat(JNIEnv *env, jobject, jlong handle, jbyteArray messages_json,
                                                              jbyteArray tools_json, jobject callback, jint max_tokens) try {
    auto &s = *reinterpret_cast<State *>(handle);
    if (max_tokens <= 0 || max_tokens > N_CTX) return error_json(env, "Invalid token limit");
    const auto should_continue = env->GetMethodID(env->GetObjectClass(callback), "shouldContinue", "()Z");
    // Every request starts from an empty context and gets the full chat
    llama_memory_clear(llama_get_memory(s.ctx), true);
    const llama_vocab *vocab = llama_model_get_vocab(s.model);

    // The model's chat template writes the tool list into the prompt, and the
    // session parses tool calls back out of the generated text
    common_chat_templates_inputs inputs;
    try {
        inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(from_bytes(env, messages_json)));
        inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(from_bytes(env, tools_json)));
    } catch (const std::exception &e) {
        return error_json(env, e.what());
    }
    inputs.parallel_tool_calls = true;
    inputs.add_bos = llama_vocab_get_add_bos(vocab);
    common_chat_session session(s.templates.get(), vocab, inputs);

    // Process the prompt in batches
    llama_tokens tokens = common_tokenize(s.ctx, session.prompt().str(), true, true);
    if ((int) tokens.size() >= N_CTX) return error_json(env, "Prompt is too long");
    common_batch batch(s.ctx);
    for (size_t i = 0; i < tokens.size(); i += N_BATCH) {
        if (!env->CallBooleanMethod(callback, should_continue)) return error_json(env, "Generation cancelled");
        batch.clear();
        const size_t end = std::min(tokens.size(), i + N_BATCH);
        for (size_t j = i; j < end; ++j) batch.add(tokens[j], (llama_pos) j, 0, j == tokens.size() - 1);
        if (llama_process(s.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            return error_json(env, "Prompt processing failed");
        }
    }
    llama_pos n_past = (llama_pos) tokens.size();

    // Sampling parameters from the LFM2.5-1.2B-Instruct model card. The session
    // adds a grammar that keeps tool calls well-formed once the model starts one.
    common_params_sampling sparams;
    sparams.temp = 0.1f;
    sparams.top_k = 50;
    sparams.top_p = 1.0f;
    sparams.min_p = 0.0f;
    sparams.penalty_repeat = 1.05f;
    session.apply_sampling(sparams);
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler_owner(
        common_sampler_init(s.model, sparams), common_sampler_free);
    auto *sampler = sampler_owner.get();
    if (!sampler) return error_json(env, "Failed to create the sampler");

    // Generate until the end of the assistant turn
    common_chat_input output;
    bool completed = false;
    for (int i = 0; i < max_tokens && n_past < N_CTX; ++i) {
        if (!env->CallBooleanMethod(callback, should_continue) || env->ExceptionCheck()) {
            return error_json(env, "Generation cancelled");
        }
        llama_token token = common_sampler_sample(sampler, s.ctx, -1);
        common_sampler_accept(sampler, token, true);
        if (llama_vocab_is_eog(vocab, token)) { completed = true; break; }

        // Keep special tokens such as <|tool_call_start|> so the parser sees them
        output.append(common_token_to_piece(s.ctx, token, true), token);
        const auto &stops = session.additional_stops();
        if (std::any_of(stops.begin(), stops.end(),
                        [&](const std::string &stop) { return string_ends_with(output.text, stop); })) {
            completed = true;
            break;
        }

        batch.clear();
        batch.add(token, n_past++, 0, true);
        if (llama_process(s.ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            return error_json(env, "Generation failed");
        }
    }

    if (!completed) return error_json(env, "Generation reached its token or context limit before completing");
    // Parse the generated text into content and tool calls
    try {
        auto msg = session.finish(output);
        // Older GGUF templates render only content, dropping structured tool-call history.
        // Preserve the generated call text for those templates; parsing still belongs to llama.cpp.
        if (!msg.tool_calls.empty() &&
            !common_chat_templates_get_caps(s.templates.get()).value("supports_tool_calls", false)) {
            msg.content = output.text;
        }
        static std::atomic<uint64_t> next_call{0};
        for (auto &call : msg.tool_calls) {
            if (call.id.empty()) call.id = "call_" + std::to_string(next_call.fetch_add(1));
        }
        return to_bytes(env, msg.to_json_oaicompat().dump());
    } catch (const std::exception &e) {
        return error_json(env, e.what());
    }
}
catch (const std::exception &error) { return error_json(env, error.what()); }

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_android_runtime_NativeAgentModel_nativeUnload(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<State *>(handle);
}
