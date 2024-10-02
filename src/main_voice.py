import asyncio
import langchain
import logging
import nest_asyncio
import numpy as np
import time
import threading
import whisper
import sounddevice as sd

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama, LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from queue import Queue
from rich.console import Console
from text_to_speech import TextToSpeech
from prompt_utils import get_session_history


console = Console()
speech_to_text = whisper.load_model("base.en")
text_to_speech = TextToSpeech()
langchain.debug = False

template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim
to provide concise responses with as little words as possible. The conversation
transcript is as follows: {history} And here is the user's follow-up: {input}
Your response:
"""

system_msg = """
    You are a helpful and friendly AI assistant. You are polite, respectful, and aim
    to provide concise responses with as little words as possible.
    """
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_msg),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
model = ChatOllama(model="gemma:2b")
parser = StrOutputParser()
chain = prompt | model | parser

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def record_audio(stop_event: threading.Event, data_queue: Queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Args:
        stop_event: An event that, when set, signals the function to stop recording.
        data_queue: A queue to which the recorded audio data will be added.
    Returns:
        None
    """

    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    Args:
        audio_np: The audio data to be transcribed.
    Returns:
        str: The transcribed text.
    """
    result = speech_to_text.transcribe(
        audio_np, fp16=False
    )  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using a language model.
    Args:
        text: The input text to be processed.
    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


def play_audio(sample_rate: int, audio_array: np.array):
    """
    Plays the given audio data using the sounddevice library.
    Args:
        sample_rate: The sample rate of the audio data.
        audio_array: The audio data to be played.
    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


async def run_chain(text: str) -> float:
    """Runs the chain as an async generator.

    This allows to print the output of the chain as it is generated, mimicking a
    lower latency.

    Args:
        text: input prompt
    """
    before_first_token = time.perf_counter()
    chunk_counter = 0
    async for chunk in chain_with_message_history.astream(
        {"input": text}, config={"configurable": {"session_id": "abc123"}}
    ):
        if chunk_counter == 0:
            after_first_token = time.perf_counter()
            chunk_counter += 1
        print(chunk, end="", flush=True)
    return after_first_token - before_first_token

if __name__ == "__main__":
    # Inserted logging only in the following lines
    logging.basicConfig(filename="assistant.log", level=logging.WARNING)
    logger = logging.getLogger(__name__)

    timings = True
    verbosity = False
    timing = {}

    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

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
                    # Timing the transcription.
                    transcribe_start_time = time.time()
                    text = transcribe(audio_np)
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

                # Timing the response generation.
                query_time_start = time.time()
                time_to_first_token = asyncio.run(run_chain(text))
                console.print()
                query_time_end = time.time()

                timing["query"] = query_time_end - query_time_start
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
