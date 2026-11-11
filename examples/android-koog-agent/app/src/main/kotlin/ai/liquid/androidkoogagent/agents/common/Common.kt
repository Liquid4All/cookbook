package ai.liquid.androidkoogagent.agents.common

import ai.liquid.androidkoogagent.App
import java.io.File

val modelPath: File
    get() = File(App.context.getExternalFilesDir(null), "LFM3-1.7B-Q4_K_M.gguf")
