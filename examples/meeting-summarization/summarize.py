"""Meeting summarization CLI."""

# /// script
# dependencies = [
#   "llama-cpp-python>=0.3.16",
# ]
# ///

from llama_cpp import Llama

def main(
    model: str,
    transcript: str,
):
    print(f"Meeting summarization CLI")

    # Load model
    model = Llama(
        model_path=model,
        n_ctx=8192,  # Increased context window to handle longer transcripts
        n_threads=4,
        verbose=False,
    )

    # Run inference
    print(f"\n{'='*80}")
    print(f"Processing transcript")
    print(f"{'='*80}\n")

    try:
        # Reset the model state before each inference to clear KV cache
        model.reset()

        # Generate summary using the model with streaming
        print("SUMMARY:")

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
        for chunk in stream:
            token = chunk['choices'][0]['text']
            print(token, end='', flush=True)

        print(f"\n\n{'='*80}\n")

    except Exception as e:
        print(f"ERROR processing transcript: {e}\n")
        print(f"{'='*80}\n")


if __name__ == "__main__":

    # CLI argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Meeting Summarization CLI - Summarize meeting transcripts using LLM")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model file")
    parser.add_argument("--transcript", type=str, required=True, help="Path to the text file containing the transcript")
    args = parser.parse_args()

    with open(args.transcript, "r") as f:
        transcript = f.read()

    main(
        args.model,
        transcript,
    )
