# Meeting summarization CLI

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

This example is a 100% local meeting summarization tool, that runs on your machine thanks to:

- `LiquidAI/LFM2-2.6B-Transcript` -> a small language model specialized in summarizing meeting transcripts.

- `llama.cpp` -> a fast inference engine with a minimal setup and state-of-the-art performance on a wide range of hardware, both locally and in the cloud.

This tool can be piped with an audio transcription model to map

- audio files to their corresponding transcripts, and
- transform the transcripts into summaries using this tool.

This 2-step pipeline can be run on your machine, without any cloud services or API keys.

Isn't that beautiful?

## Quick start

To run this CLI tool, you need to have `uv` installed in your system. Follow instructions [here](https://docs.astral.sh/uv/getting-started/installation/) to install it.

Once you have `uv` installed, you can run the tool with `uv run` without cloning the repository.
```sh
TODO
```

Alternatively, if you plan on digging deeper and modifying the code, you can clone the repository

```sh
git clone https://github.com/Liquid4All/cookbook.git
cd cookbook/examples/meeting-summarization
```

and run the summarization CLI using the following command:

```sh
uv run summarize.py \
  --model LiquidAI/LFM2-2.6B-Transcript-GGUF \
  --hf-model-file LFM2-2.6B-Transcript-1-GGUF.gguf \
  --transcript transcript.txt
```

## How does it work?

The CLI uses the llama.cpp Python bindings to download the llama.cpp binary for your platform automatically, so you don't need to worry about it.

Then, it uses the `LiquidAI/LFM2-2.6B-Transcript` model to summarize the transcript.

```python
model = Llama(
    model_path="LiquidAI/LFM2-2.6B-Transcript-GGUF",
    n_ctx=8192,
    n_threads=4,
    verbose=False,
)
```

