# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an invoice extraction tool that processes bill/invoice images using Large Foundation Models (LFMs). It extracts key information (bill type and amount) from uploaded images and appends the data to a CSV file for expense tracking.

## Development Commands

### Running the Application
```bash
# Basic usage pattern
uv run python src/agentic_workflow_for_laptop \
    --dir path/to/dir/where/user/saves/bill/pictures \
    --output path/to/csv/file/with/parsed/data \
    --extractor-model name_of_lfm2_model_used_for_data_extraction \
    --image-process-model name_of_lfm2_vl_model_used_for_image_parsing
```

### Recommended Models
- **Image Processing**: [LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B) for converting images to text
- **Data Extraction**: [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract) for parsing text into structured data

## Architecture

### Project Structure
```
src/agentic_workflow_for_laptop/
├── __init__.py          # Package initialization with hello() function
├── main.py              # Empty main module (implementation needed)
└── py.typed             # Type hint marker file
```

### Core Workflow
1. **File Watching**: The CLI continuously monitors the input directory for new image files
2. **Image Processing**: Uses Ollama Python SDK with LFM2-VL model to extract text from images
3. **Data Extraction**: Processes extracted text with LFM2 model to identify bill type and amount
4. **CSV Output**: Appends structured data to the specified CSV file

### Dependencies
- **Package Manager**: Uses `uv` for Python package management
- **Inference Engine**: Ollama Python SDK for LFM model inference
- **Python Version**: Requires Python >=3.12
- **Build System**: Hatchling

## Technical Implementation Notes

The project uses the Ollama Python SDK pattern for image processing:
```python
import ollama

response = ollama.chat(
    model='model_name',
    messages=[{
        'role': 'user',
        'content': 'Describe this image.',
        'images': [image_path]
    }]
)
```

The application is designed as a real-time file watcher that processes invoice images immediately upon detection in the monitored directory.

## Guidance

Every time you want to run a python script with `python` make sure you activate the virtual environment by using `uv run python`