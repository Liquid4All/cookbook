# Invoice Extractor Tool

## What's this example?
This is an automated invoice processing tool that monitors a folder for new invoice files and automatically extracts key information from them.

When you drop an invoice photo into a watched directory, the tool uses a chain with 2 language models:

- [LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) -> local Vision Language Model to extract a textual description of the invoice

- [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract) -> local unstructured-to-structured text-to-text Language Model that tranforms the raw text into a structured record. This record is appended to a CSV file.

![](./media/chain_diagram.gif)

This a practical example of building agentic workflows that run entirely on your local machine: no API keys, no cloud costs, no private data shared with third-parties.

## Environment setup

You will need

- [Ollama](https://ollama.com/) to serve the Language Models locally.
- [uv](https://docs.astral.sh/uv/) to manage Python dependencies and run the application efficiently without creating virtual environments manually.

### Install Ollama

<details>
<summary>Click to see installation instructions for your platform</summary>

**macOS:**
```bash
# Download and install from the website
# Visit: https://ollama.ai/download

# Or use Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from [https://ollama.ai/download](https://ollama.ai/download)

</details>


### Install UV

<details>
<summary>Click to see installation instructions for your platform</summary>

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</details>


## How to run it?

For example:
```sh
uv run python src/invoice_parser/main.py \
    --dir invoices/ \
    --image-model hf.co/LiquidAI/LFM2-VL-3B-GGUF:F16 \
    --extractor-model hf.co/LiquidAI/LFM2-1.2B-Extract-GGUF:F16 \
    --process-existing
```

## Further improvements

Here is a list of features I challenge you to implement:

- [ ] Add a system prompt in the VLM to identify and discard images that do not correspond to bills.
- [ ] Add an on_delete handler to keep in sync the list of invoices in the directory and the `bills.csv` file.


## Wanna learn more about building prod-ready local agentic workflows?

<a href="https://discord.gg/DFU3WQeaYD"><img src="https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white" alt="Join Discord"></a></a>

