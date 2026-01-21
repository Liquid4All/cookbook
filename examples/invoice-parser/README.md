# Invoice extractor tool

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

A Python CLI that extracts payment details from invoice pdfs.

This a practical example of building local AI tools and apps with

- No cloud costs
- No network latency
- No data privacy loss


## What's inside?

In this example, you will learn how to:

- **Set up local AI inference** using Ollama to run Liquid models entirely on your machine without requiring cloud services or API keys
- **Build a file monitoring system** that automatically processes new files dropped into a directory
- **Extract structured output from images** using LFM2-VL-3B, a small vision-language model.


## Understanding the architecture

When you drop an invoice photo into a watched directory, the tool uses [LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) to extract a structured record with the main information in the invoice, including

- The utility to pay
- The amount to pay
- The currency of the payment.

This record is appended to a CSV file.

![](./media/diagram.jpg)


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

Let's start by cloning the repository:

```sh
git clone https://github.com/Liquid4All/cookbook.git
cd cookbook/examples/invoice-parser
```

Then, run the application using the invoices that are already in the repository:
```sh
uv run python src/invoice_parser/main.py \
    --dir invoices/ \
    --image-model hf.co/LiquidAI/LFM2-VL-3B-GGUF:F16 \
    --process-existing
```


Feel free to modify the path to the invoices directory and the model IDs to suit your needs.

> [!NOTE]
>
> You can use the 1.6B version of the VLM model as follows:
> 
> ```sh
> uv run python src/invoice_parser/main.py \
>     --dir invoices/ \
>     --image-model hf.co/LiquidAI/LFM2-VL-1.6B-GGUF:F16 \
>     --process-existing
> ```

If you have `make` installed, you can run the application with the following command:
```sh
make run
```

The data extracted from the invoices is be saved in the same directory as the invoices, in a file called `bills.csv`.
If you open the file, you will see the following data:

| processed_at | file_path | utility | amount | currency |
|--------------|-----------|---------|--------|----------|
| 2025-10-31 11:25:47 | invoices/water_australia.png | electricity | 68.46 | AUD |
| 2025-10-31 11:26:00 | invoices/Sample-electric-Bill-2023.jpg | electricity | 28.32 | USD |
| 2025-10-31 11:26:09 | invoices/british_gas.png | electricity | 81.31 | GBP |
| 2025-10-31 11:42:35 | invoices/castlewater1.png | electricity | 150.0 | USD |

Observations:
- The first 3 invoices are properly extracted, with the correct amount and currency.
- The fourth invoice is not properly extracted, where both amount and currency are not correct.

## Next steps

We have a tool that works well 75% of the time on our sample of invoices, which is

- good enough for a demo
- not good enough for a production-ready application

As a next step, before diving into prompt optimizations or fine-tuning, we will try to use the latest `LFM2.5-VL-1.6B` and compare results.

If you are interested in learning more about model customization for Vision Language Models, we recommend you to check out the following example:

- [Cats vs dogs identification from images](https://github.com/Paulescu/image-classification-with-local-vlms/tree/main)


## Need help?

Join the [Liquid AI Discord Community](https://discord.com/invite/liquid-ai) and ask.
[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

