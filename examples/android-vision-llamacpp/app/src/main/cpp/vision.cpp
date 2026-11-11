#include <jni.h>
#include <algorithm>
#include <string>
#include <unistd.h>
#include "llama.h"
#include "ggml-backend.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"

static std::string run(const std::string &model_path, const std::string &projector_path,
                       const std::string &image_path, const std::string &question) {
    llama_model *model = llama_model_load_from_file(model_path.c_str(), llama_model_default_params());
    if (!model) return "Could not load the text model.";
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = std::max(2, std::min(4, (int)sysconf(_SC_NPROCESSORS_ONLN) - 2));
    cp.n_threads_batch = cp.n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return "Could not create the inference context."; }
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = false;
    mp.n_threads = cp.n_threads;
    mtmd_context *vision = mtmd_init_from_file(projector_path.c_str(), model, mp);
    if (!vision) { llama_free(ctx); llama_model_free(model); return "Could not load the vision projector."; }
    auto wrapper = mtmd_helper_bitmap_init_from_file(vision, image_path.c_str(), false, mtmd_helper_init_opt_default());
    if (!wrapper.bitmap) { mtmd_free(vision); llama_free(ctx); llama_model_free(model); return "Could not read the image."; }
    auto templates = common_chat_templates_init(model, "");
    common_chat_msg msg; msg.role = "user"; msg.content = std::string(mtmd_default_marker()) + "\n" + question;
    std::vector<common_chat_msg> history;
    std::string prompt = common_chat_format_single(templates.get(), history, msg, true, false);
    mtmd_input_text input = {prompt.data(), prompt.size(), true, true};
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const mtmd_bitmap *bitmaps[] = {wrapper.bitmap};
    std::string result;
    int rc = mtmd_tokenize(vision, chunks, &input, bitmaps, 1);
    llama_pos n_past = 0;
    if (rc == 0) rc = mtmd_helper_eval_chunks(vision, ctx, chunks, 0, 0, cp.n_batch, true, &n_past);
    if (rc != 0) result = "Image processing failed (code " + std::to_string(rc) + ").";
    else {
        common_params_sampling sp; sp.temp = 0.1f; sp.top_k = 50; sp.penalty_repeat = 1.1f;
        common_sampler *sampler = common_sampler_init(model, sp);
        common_batch batch(ctx);
        const llama_vocab *vocab = llama_model_get_vocab(model);
        for (int i = 0; i < 256 && n_past < (llama_pos)cp.n_ctx - 1; ++i) {
            llama_token token = common_sampler_sample(sampler, ctx, -1);
            common_sampler_accept(sampler, token, true);
            if (llama_vocab_is_eog(vocab, token)) break;
            result += common_token_to_piece(ctx, token);
            batch.clear(); batch.add(token, n_past++, 0, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                result += "\nGeneration stopped due to a decoding error."; break;
            }
        }
        common_sampler_free(sampler);
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(wrapper.bitmap);
    if (wrapper.video_ctx) mtmd_helper_video_free(wrapper.video_ctx);
    mtmd_free(vision);
    llama_free(ctx);
    llama_model_free(model);
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_liquid_visiondemo_VisionEngine_infer(JNIEnv *env, jobject, jstring model,
                                              jstring projector, jstring image, jstring question) {
    auto get = [env](jstring s) { const char *p = env->GetStringUTFChars(s, nullptr); std::string out(p); env->ReleaseStringUTFChars(s, p); return out; };
    std::string result = run(get(model), get(projector), get(image), get(question));
    return env->NewStringUTF(result.c_str());
}
extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_visiondemo_VisionEngine_init(JNIEnv *env, jobject, jstring native_dir) {
    const char *path = env->GetStringUTFChars(native_dir, nullptr);
    ggml_backend_load_all_from_path(path);
    env->ReleaseStringUTFChars(native_dir, path);
    llama_backend_init();
}
