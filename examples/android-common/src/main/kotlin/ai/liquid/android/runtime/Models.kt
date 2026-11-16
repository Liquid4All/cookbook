package ai.liquid.android.runtime

/** Sizes and SHA-256 values come from the repositories' Hugging Face LFS metadata. */
object Models {
    val text = ModelSpec("LFM2.5-1.2B-Instruct-Q5_K_M.gguf",
        "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q5_K_M.gguf",
        843354944, "fa03f3ac4da941a53a0cd4450aacf6a80804c6a1ff885d2fdcbe9406c03215c4")
    val small = ModelSpec("LFM2.5-350M-Q8_0.gguf",
        "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/657e078c94084481950a2d555a941481f715536b/LFM2.5-350M-Q8_0.gguf",
        379217632, "be036a757295e550098b85e13f6af2735d0fa73b41e1156a40c7d8e8e32a5766")
    val vision = ModelSpec("LFM2.5-VL-1.6B-Q8_0.gguf",
        "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/36fc16bc95133424921bcc3da009e83b2f23ffb5/LFM2.5-VL-1.6B-Q8_0.gguf",
        1246254880, "a34bd1506a298d7ff07902e69baeac48c7c20bb85162e61218b743dc10be7c67")
    val projector = ModelSpec("mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
        "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/36fc16bc95133424921bcc3da009e83b2f23ffb5/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
        583109888, "2ce89e610c56f3198ece2b86cf61743a08b9307279c89125eb2412ebb908689d")
}
