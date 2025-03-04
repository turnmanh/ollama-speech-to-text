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
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama, LlamaCpp
from langchain_ollama import ChatOllama
from queue import Queue
from rich.console import Console
from text_to_speech import TextToSpeech


console = Console()
speech_to_text = whisper.load_model("base.en")
text_to_speech = TextToSpeech()
langchain.debug = False

# template = """
# You are a helpful and friendly AI assistant. You are polite, respectful, and aim
# to provide concise responses with as little words as possible. The conversation
# transcript is as follows: {history} And here is the user's follow-up: {input}
# Your response:
# """


template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim
to provide concise responses with as little words as possible. Here is the
user's follow-up: {input} Your response:
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOllama(model="gemma:2b")
parser = StrOutputParser()

chain = prompt | model | parser

# PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# chain = ConversationChain(
#     prompt=PROMPT,
#     verbose=False,
#     memory=ConversationBufferMemory(ai_prefix="Aissistant:"),
#     llm=Ollama(
#         model="llama3.2:1b"
#     ),
# llm=LlamaCpp(
#     model_path="/home/maternush/Downloads/llama-2-7b-chat.Q5_K_M.gguf",
#     temperature=0.75,
#     max_tokens=2000,
#     top_p=1,
#     verbose=False,
# ),
# )


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


async def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using a language model.
    Args:
        text: The input text to be processed.
    Returns:
        str: The generated response.
    """
    # response = chain.predict(input=text)
    # response = ""
    async for chunk in chain.astream(input=text):
        print(chunk, end="|", flush=True)
        # response += chunk + " "
    # if response.startswith("Assistant:"):
    #     response = response[len("Assistant:") :].strip()
    # return response
    return "done"


async def run_chain(text: str, console: Console):
    async for chunk in chain.astream(input=text):
        print(chunk, end="", flush=True)


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


if __name__ == "__main__":
    # Inserted logging only in the following lines
    logging.basicConfig(filename="assistant.log", level=logging.WARNING)
    logger = logging.getLogger(__name__)
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    nest_asyncio.apply()
    try:
        while True:
            # asyncio.set_event_loop(asyncio.new_event_loop())

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
                    text = transcribe(audio_np)
                    logger.info(f"Transcribed text from audio: {text}")
                console.print(f"[yellow]You: {text}")

                # with console.status("Generating response...",
                # spinner="earth"):
                console.rule("[cyan]Response")
                asyncio.run(run_chain(text, console=console))
                console.print()
                console.rule()
                #     response = get_llm_response(text)
                # logger.info(f"Generated response: {response}")
                # sample_rate, audio_array = text_to_speech.synthesize_long_text(
                #     response
                # )

                # console.print(f"[cyan]Assistant: {response}")
                # play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
