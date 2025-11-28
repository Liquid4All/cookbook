import torch
import torchaudio
from pathlib import Path
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .audio_file_manager import AudioFileManager
from news_reader.audio_recorder import AudioRecorder
import time

app = get_modal_app("voice-chat-example")
image = get_docker_image()
volume = get_volume("voice-chat-example-volume")


# TODOs
# - Add modal decorator
# - Attach volume
# - Run
@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/assets": volume,
    },
    secrets=get_secrets(),
    timeout=1 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def run(session_id: str):
    # Load models
    HF_REPO = "LiquidAI/LFM2-Audio-1.5B"

    processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
    model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

    # Set up inputs for the model
    chat = ChatState(processor)

    # System prompt
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()

    # User prompt with audio input
    chat.new_turn("user")
    wav, sampling_rate = torchaudio.load(f"/assets/{session_id}/question.wav")
    chat.add_audio(wav, sampling_rate)
    chat.end_turn()

    chat.new_turn("assistant")

    # Generate text and audio tokens.
    text_out: list[torch.Tensor] = []
    audio_out: list[torch.Tensor] = []
    modality_out: list[LFMModality] = []
    for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
        if t.numel() == 1:
            print(processor.text.decode(t), end="", flush=True)
            text_out.append(t)
            modality_out.append(LFMModality.TEXT)
        else:
            audio_out.append(t)
            modality_out.append(LFMModality.AUDIO_OUT)

    # output: Sure! How about "Handcrafted Woodworking, Precision Made for You"? Another option could be "Quality Woodworking, Quality Results." If you want something more personal, you might try "Your Woodworking Needs, Our Expertise."

    # Detokenize audio, removing the last "end-of-audio" codes
    # Mimi returns audio at 24kHz
    mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
    with torch.no_grad():
        waveform = processor.mimi.decode(mimi_codes)[0]
    torchaudio.save(f"/assets/{session_id}/answer1.wav", waveform.cpu(), 24_000)

    # # Append newly generated tokens to chat history
    # chat.append(
    #     text = torch.stack(text_out, 1),
    #     audio_out = torch.stack(audio_out, 1),
    #     modality_flag = torch.tensor(modality_out),
    # )
    # chat.end_turn()

    # # Start new turn
    # chat.new_turn("user")
    # chat.add_text("My business specialized in chairs, can you give me something related to that?")
    # chat.end_turn()

    # chat.new_turn("assistant")

    # # Generate second turn text and audio tokens.
    # audio_out: list[torch.Tensor] = []
    # for t in model.generate_interleaved(**chat, max_new_tokens=512, audio_temperature=1.0, audio_top_k=4):
    #     if t.numel() == 1:
    #         print(processor.text.decode(t), end="", flush=True)
    #     else:
    #         audio_out.append(t)

    # # output: Sure thing! How about “Comfortable Chairs, Crafted with Care” or “Elegant Seats, Handcrafted for You”? Let me know if you’d like a few more options.

    # # Detokenize second turn audio, removing the last "end-of-audio" codes
    # mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
    # with torch.no_grad():
    #     waveform = processor.mimi.decode(mimi_codes)[0]
    # torchaudio.save("/assets/answer2.wav", waveform.cpu(), 24_000)

def generate_session_id() -> str:
    """
    Generate a unique session ID based on the current timestamp.
    """
    import datetime
    now = datetime.datetime.now()
    session_id = now.strftime("%Y%m%d_%H%M%S")
    return session_id

def record() -> Path:
    """
    Records from the microphone and saves to a local wav file.

    Returns:
        Path to the saved wav file.
    """
    import numpy as np
    
    recorder = AudioRecorder(sample_rate=16000, channels=1)

    print("Starting recording... Speak now!")
    
    # Start recording with auto-stop on silence and rich audio bar
    recorder.start_recording(
        silence_duration=2.0,    # Stop after 2 seconds of silence
        silence_threshold=0.02,  # Higher threshold = less sensitive to background noise
        show_audio_bar=True      # Enable rich audio visualization
    )

    # Wait for recording to finish automatically
    while recorder.is_recording:
        time.sleep(0.1)

    print("\nRecording finished, processing...")

    # Get the recorded audio
    audio_data = recorder.stop_recording()
    
    print(f"Audio data length: {len(audio_data)} samples")

    # Save it
    filename = "user_recording.wav"
    recorder.save_to_file(filename, audio_data)
    recorder.cleanup()
    
    # Return absolute path to the file
    return Path(filename).resolve()

@app.local_entrypoint()
def local_entrypoint():

    # generate a unique session id
    session_id = generate_session_id()

    # record user question
    audio_file = record()

    # initialize audio file manager and push audio file to modal volume
    audio_manager = AudioFileManager("voice-chat-example-volume", session_id)
    
    # upload audio file (automatically goes to session directory)
    audio_manager.upload(str(audio_file))
    
    # run remote function 
    run.remote(session_id)
    
    # download the generated answer
    local_answer_path = f"answer_{session_id}.wav"
    audio_manager.download("answer1.wav", local_answer_path)
    
    print(f"Generated audio saved to: {local_answer_path}")

if __name__ == "__main__":

    local_entrypoint()

