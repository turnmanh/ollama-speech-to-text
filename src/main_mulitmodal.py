import asyncio
import logging
import nest_asyncio
import numpy as np
import time
import threading
import whisper

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from PIL import Image
from queue import Queue
from rich.console import Console

from image_utils import convert_to_base64, prep_image
from prompt_utils import get_prompt, get_prompt_with_history, get_session_history
from voice_utils import record_audio, transcribe


async def run_chain(chain, data: dict) -> float:
    """Runs the chain as an async generator.

    This allows to print the output of the chain as it is generated, mimicking a
    lower latency.

    Args:
        text: input prompt
    Returns:
        float: time taken to generate the first token
    """
    before_first_token = time.perf_counter()
    chunk_counter = 0
    async for chunk in chain.astream(
        # {"text": input_text, "image": image_b64}
        {"input": data}, config={"configurable": {"session_id": "abc123"}},
        ):
        if chunk_counter == 0:
            after_first_token = time.perf_counter()
            chunk_counter += 1
        print(chunk, end="", flush=True)
    return after_first_token - before_first_token


def main(
    chain,
    image_b64: str,
    console: Console,
    speech_to_text: whisper.Whisper,
    timings: bool = False,
):
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    timing = {}

    nest_asyncio.apply()

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

                console.rule("[cyan]Assistant")
                # Time the response generation process.
                query_start_time = time.time()
                time_to_first_token = asyncio.run(
                    run_chain(chain, {"image":image_b64, "text":text})
                )
                console.print()
                query_end_time = time.time()

                timing["query"] = query_end_time - query_start_time
                timing["time_to_first_token"] = time_to_first_token

                # logger.info(f"Generated response: {response}")
                # sample_rate, audio_array = text_to_speech.synthesize_long_text(
                #     response
                # )

                if timings:
                    console.rule("[cyan]Timings")
                    console.print(
                        ":clock8:",
                        f"[bold]Transcription[/bold] in {timing['transcribe']:.2f}s.",
                        f"[bold]Time to first token[/bold] {timing['time_to_first_token']:.2f}s.",
                        f"[bold]Total Query[/bold] in {timing['query']:.2f}s.",
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
    model = ChatOllama(model="moondream")
    parser = StrOutputParser()

    # Create a chain of functions to process the query.
    # chain = get_prompt | model | parser
    chain = get_prompt_with_history | model | parser

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    main(
        # chain=chain,
        chain=chain_with_message_history,
        image_b64=image_b64,
        console=console,
        speech_to_text=speech_to_text,
        timings=True,
    )
