from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from rich.console import Console

console = Console()


def load_model(hf_repo_id: str, hf_model_file: str) -> Llama:
    """
    Load a Llama model from either a local GGUF file or a HuggingFace repository.

    Args:
        hf_repo_id: Either a local path to a GGUF file, or a HuggingFace repository ID
        hf_model_file: If provided, treats 'model' as a HF repo ID and downloads this specific GGUF file

    Returns:
        Llama: Loaded model instance

    Examples:
        # Load from local path
        model = load_model("path/to/model.gguf")

        # Load from HuggingFace
        model = load_model("TheBloke/Llama-2-7B-GGUF", hf_model_file="llama-2-7b.Q4_K_M.gguf")
    """
    # Download from HuggingFace if the model is not already cached locally
    console.print(
        f"[cyan]Downloading[/cyan] [yellow]{hf_model_file}[/yellow] from HuggingFace repository: [blue]{hf_repo_id}[/blue]"
    )
    model_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=hf_model_file,
    )
    console.print(f"[green]✓[/green] Model file is at: [dim]{model_path}[/dim]")

    # Load the model
    console.print(
        f"[cyan]Loading model from local path:[/cyan] [dim]{model_path}[/dim]"
    )
    model = Llama(
        model_path=model_path,
        n_ctx=32768,  # Increased context window to handle longer transcripts
        n_threads=4,
        verbose=False,
    )

    return model
