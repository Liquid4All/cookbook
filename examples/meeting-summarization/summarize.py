"""Meeting summarization CLI."""

# /// script
# dependencies = [
#   "llama-cpp-python>=0.3.16",
#   "huggingface-hub>=0.20.0",
#   "rich>=13.0.0",
# ]
# ///

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def load_model(model: str, hf_model_file: str = None) -> Llama:
    """
    Load a Llama model from either a local GGUF file or a HuggingFace repository.

    Args:
        model: Either a local path to a GGUF file, or a HuggingFace repository ID
        hf_model_file: If provided, treats 'model' as a HF repo ID and downloads this specific GGUF file

    Returns:
        Llama: Loaded model instance

    Examples:
        # Load from local path
        model = load_model("path/to/model.gguf")

        # Load from HuggingFace
        model = load_model("TheBloke/Llama-2-7B-GGUF", hf_model_file="llama-2-7b.Q4_K_M.gguf")
    """
    if hf_model_file:
        # Download from HuggingFace
        console.print(f"[cyan]Downloading[/cyan] [yellow]{hf_model_file}[/yellow] from HuggingFace repository: [blue]{model}[/blue]")
        model_path = hf_hub_download(
            repo_id=model,
            filename=hf_model_file,
        )
        console.print(f"[green]✓[/green] Model downloaded to: [dim]{model_path}[/dim]")
    else:
        # Use local path
        model_path = model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        console.print(f"[cyan]Loading model from local path:[/cyan] [dim]{model_path}[/dim]")

    # Load the model
    model = Llama(
        model_path=model_path,
        n_ctx=8192,  # Increased context window to handle longer transcripts
        n_threads=4,
        verbose=False,
    )

    return model


def main(
    model: str,
    transcript: str,
    hf_model_file: str = None,
):
    console.print(Panel.fit(
        "[bold cyan]Meeting Summarization CLI[/bold cyan]",
        border_style="cyan"
    ))

    # Load model
    model = load_model(model, hf_model_file)

    # Run inference
    console.print(Panel(
        "[bold yellow]Processing transcript...[/bold yellow]",
        border_style="yellow"
    ))

    try:
        # Reset the model state before each inference to clear KV cache
        model.reset()

        # Generate summary using the model with streaming
        console.print("\n[bold green]SUMMARY:[/bold green]\n")

        system_prompt = """
        Provide a comprehensive summary of the transcript, broken down as indicated below.

        - TOPICS_DISCUSSED: Enumerate the primary discussion points contained within this meeting record
        - EXECUTIVE_SUMMARY: Provide a concise overview – two or three sentences – outlining the primary conclusions and resolutions detailed in the transcript.
        - KEY_DECISIONS: Identify and enumerate the definitive decisions reached during this meeting. Focus specifically on outcomes and actions agreed upon. A straightforward list will suffice.

        Organize the response with distinct headings delineating each requested summary type.
        """

        stream = model(
            system_prompt + "\n\n" + transcript,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stream=True,  # Enable streaming
        )

        # Collect tokens and print as they arrive
        summary_text = ""
        for chunk in stream:
            token = chunk['choices'][0]['text']
            summary_text += token
            console.print(token, end='', highlight=False)

        console.print("\n")
        console.rule("[dim]End of Summary[/dim]", style="dim")

    except Exception as e:
        console.print(Panel(
            f"[bold red]ERROR processing transcript:[/bold red]\n{e}",
            border_style="red",
            title="Error"
        ))


if __name__ == "__main__":

    # CLI argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Meeting Summarization CLI - Summarize meeting transcripts using LLM")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to local GGUF model file, or HuggingFace repository ID (e.g., 'TheBloke/Llama-2-7B-GGUF')"
    )
    parser.add_argument(
        "--hf-model-file",
        type=str,
        default=None,
        help="If using HuggingFace, specify the GGUF filename within the repo (e.g., 'llama-2-7b.Q4_K_M.gguf')"
    )
    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Path to the text file containing the transcript"
    )
    args = parser.parse_args()

    with open(args.transcript, "r") as f:
        transcript = f.read()

    main(
        args.model,
        transcript,
        args.hf_model_file,
    )
