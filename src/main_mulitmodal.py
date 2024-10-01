import logging
import numpy as np
import threading
import whisper

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from PIL import Image
from queue import Queue
from rich.console import Console

from image_utils import convert_to_base64, prep_image
from prompt_utils import get_prompt
from voice_utils import record_audio, transcribe


def main(chain, image_b64: str, console: Console, speech_to_text: whisper.Whisper):
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np, speech_to_text=speech_to_text)

                    # Fallback text if no transcription is available.
                    if len(text) == 0:
                        text = "Describe the image to me."

                    logger.info(f"Transcribed text from audio: {text}")
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = chain.invoke({"text": text, "image": image_b64})
                    logger.info(f"Generated response: {response}")
                    # sample_rate, audio_array = text_to_speech.synthesize_long_text(
                    #     response
                    # )

                console.print(f"[cyan]Assistant: {response}")
                # play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")


if __name__ == "__main__":

    # Inserted logging only in the following lines
    logging.basicConfig(filename="assistant.log", level=logging.WARNING)
    logger = logging.getLogger(__name__)

    # Initialize the console and the speech recognition model.
    console = Console()
    speech_to_text = whisper.load_model("base.en")

    # Load the image and convert it to base64.
    file_path = "./data/img_castle.jpg"
    image_b64 = prep_image(image_path=file_path)

    # Initialize the model and parser.
    model = ChatOllama(model="llava-phi3")
    parser = StrOutputParser()

    # Create a chain of functions to process the query.
    chain = get_prompt | model | parser

    main(
        chain=chain, image_b64=image_b64, console=console, speech_to_text=speech_to_text
    )
