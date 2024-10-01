import logging
import numpy as np
import time
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


def main(
    chain,
    image_b64: str,
    console: Console,
    speech_to_text: whisper.Whisper,
    timings: bool = False,
):
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    timing = {}

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
                    # Time the transcription process.
                    transcribe_start_time = time.time()
                    text = transcribe(audio_np, speech_to_text=speech_to_text)
                    transcribe_end_time = time.time()
                    timing["transcribe"] = transcribe_end_time - transcribe_start_time

                    # Fallback text if no transcription is available.
                    if len(text) == 0:
                        if verbosity:
                            console.print(
                                f"[yellow]Got an input of length: {len(text)}. Using fallback text."
                            )
                        text = "Describe the image to me."

                    logger.info(f"Transcribed text from audio: {text}")
                
                console.rule("[cyan]Transcription")
                console.print(f"[yellow][bold]Me[/bold]: {text}")

                with console.status("Generating response...", spinner="earth"):
                    # Time the response generation process.
                    query_start_time = time.time()
                    response = chain.invoke({"text": text, "image": image_b64})
                    query_end_time = time.time()
                    timing["query"] = query_end_time - query_start_time

                    logger.info(f"Generated response: {response}")
                    # sample_rate, audio_array = text_to_speech.synthesize_long_text(
                    #     response
                    # )

                # Print the response separated from the rest of the output and play the audio.
                console.rule("[cyan]Assistant")
                console.print(
                    response, overflow="fold"
                )  # Folding of output on overflow.
                if timings:
                    console.rule("[cyan]Timings")
                    console.print(
                        ":clock8:",
                        f"Transcription in {timing['transcribe']:.2f}s.",
                        f"Query in {timing['query']:.2f}s.",
                        sep=" ",
                    )
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

    verbosity = False

    # Initialize the console and the speech recognition model.
    console = Console()
    speech_to_text = whisper.load_model("base.en")

    # Load the image and convert it to base64.
    file_path = "./data/img_castle.jpg"
    image_b64 = prep_image(image_path=file_path)

    # Initialize the model and parser.
    model = ChatOllama(model="llama3.2:1b")
    parser = StrOutputParser()

    # Create a chain of functions to process the query.
    chain = get_prompt | model | parser

    main(
        chain=chain,
        image_b64=image_b64,
        console=console,
        speech_to_text=speech_to_text,
        timings=True,
    )
