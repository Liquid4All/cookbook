# Invoice extraction tool

## How does it work?
You upload invoice pictures on a given folder, and the invoice extractor tool extracts key information, including:

- Bill type
- Bill amount

and adds this data to a given CSV file the user has to track spending.

## Under the hood

The user starts the tool

```
uv run python src/agentic_workflow_for_laptop \
    --dir path/to/dir/where/user/saves/bill/pictures \
    --output path/to/csv/file/with/parsed/data \
    --extractor-model name_of_lfm2_model_used_for_data_extraction \
    --image-process-model name_of_lfm2_vl_model_used_for_image_parsing
```

For example, 2 good models would be:

- [LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) to transform raw pixels into textual information
- [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract) to parse the extracted information into a (bill type, bill amount) pairs
that can be appended to the CSV file.

For the inference we use the Ollama Python SDK with image support, for example

```python
import ollama

# Example using a local image file
image_path = 'path/to/your/image.jpg'

response = ollama.chat(
    model='gemma3:4b',
    messages=[
        {
            'role': 'user',
            'content': 'Describe this image.',
            'images': [image_path] # Pass image path directly
        }
    ]
)

print(response['message']['content'])
```

The CLI is constanly watching for any new file added to the input `--dir` and processing it inmediately.


## TODOs

- [ ] Add system prompt in the VLM to identify and discard images that do not correspond to bills.
- [ ] Add a sync mechanism where the bill.csv is synced with the current state of the invoices in the directory.


