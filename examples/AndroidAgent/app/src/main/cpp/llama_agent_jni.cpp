// Minimal JNI wrapper around llama.cpp for tool calling: load a text model once,
// then turn a chat (OpenAI-format messages and tools) into the model's next
// assistant message, with any tool calls already parsed.
#include <jni.h>
#include <algorithm>
#include <string>
#include <unistd.h>

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "sampling.h"

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static common_chat_templates_ptr g_templates;

constexpr int N_CTX = 8192;
constexpr int N_BATCH = 512;
constexpr int N_PREDICT = 1024;

static std::string to_string(JNIEnv *env, jstring s) {
    const char *chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

static jstring error_json(JNIEnv *env, const std::string &message) {
    return env->NewStringUTF(common_json{{"error", message}}.dump().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_koogleapsdk_agents_common_LlamaChat_nativeInit(JNIEnv *env, jobject, jstring native_lib_dir) {
    // Load the CPU backend variants that ship as separate .so files
    ggml_backend_load_all_from_path(to_string(env, native_lib_dir).c_str());
    llama_backend_init();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_liquid_koogleapsdk_agents_common_LlamaChat_nativeLoad(JNIEnv *env, jobject, jstring model_path) {
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

extern "C" JNIEXPORT jstring JNICALL
Java_ai_liquid_koogleapsdk_agents_common_LlamaChat_nativeChat(JNIEnv *env, jobject, jstring messages_json,
                                                              jstring tools_json) {
    // Every request starts from an empty context and gets the full chat
    llama_memory_clear(llama_get_memory(g_ctx), true);
    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    // The model's chat template writes the tool list into the prompt, and the
    // session parses tool calls back out of the generated text
    common_chat_templates_inputs inputs;
    try {
        inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(to_string(env, messages_json)));
        inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(to_string(env, tools_json)));
    } catch (const std::exception &e) {
        return error_json(env, e.what());
    }
    inputs.parallel_tool_calls = true;
    inputs.add_bos = llama_vocab_get_add_bos(vocab);
    common_chat_session session(g_templates.get(), vocab, inputs);

    // Process the prompt in batches
    llama_tokens tokens = common_tokenize(g_ctx, session.prompt().str(), true, true);
    if ((int) tokens.size() >= N_CTX) return error_json(env, "Prompt is too long");
    common_batch batch(g_ctx);
    for (size_t i = 0; i < tokens.size(); i += N_BATCH) {
        batch.clear();
        const size_t end = std::min(tokens.size(), i + N_BATCH);
        for (size_t j = i; j < end; ++j) batch.add(tokens[j], (llama_pos) j, 0, j == tokens.size() - 1);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
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
    common_sampler *sampler = common_sampler_init(g_model, sparams);
    if (!sampler) return error_json(env, "Failed to create the sampler");

    // Generate until the end of the assistant turn
    common_chat_input output;
    for (int i = 0; i < N_PREDICT && n_past < N_CTX; ++i) {
        llama_token token = common_sampler_sample(sampler, g_ctx, -1);
        common_sampler_accept(sampler, token, true);
        if (llama_vocab_is_eog(vocab, token)) break;

        // Keep special tokens such as <|tool_call_start|> so the parser sees them
        output.append(common_token_to_piece(g_ctx, token, true), token);
        const auto &stops = session.additional_stops();
        if (std::any_of(stops.begin(), stops.end(),
                        [&](const std::string &stop) { return string_ends_with(output.text, stop); })) {
            break;
        }

        batch.clear();
        batch.add(token, n_past++, 0, true);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            common_sampler_free(sampler);
            return error_json(env, "Generation failed");
        }
    }
    common_sampler_free(sampler);

    // Parse the generated text into content and tool calls
    try {
        const common_chat_msg &msg = session.finish(output);
        return env->NewStringUTF(msg.to_json_oaicompat().dump().c_str());
    } catch (const std::exception &e) {
        return error_json(env, e.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_koogleapsdk_agents_common_LlamaChat_nativeUnload(JNIEnv *, jobject) {
    g_templates.reset();
    if (g_ctx) llama_free(g_ctx);
    if (g_model) llama_model_free(g_model);
    g_ctx = nullptr;
    g_model = nullptr;
}
