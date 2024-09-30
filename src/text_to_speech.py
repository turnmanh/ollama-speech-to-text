import nltk
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel
from langchain_core._api import LangChainDeprecationWarning

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=LangChainDeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class TextToSpeech:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextToSpeech class.
        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given text using the specified voice preset.
        Args:
            text: The input text to be synthesized.
            voice_preset: The voice preset to be used for the synthesis.
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=1_000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def synthesize_long_text(self, text: str, voice_preset: str = "v2/en_speaker_1"):
        """
        Synthesizes audio from the given long-form text using the specified voice preset.
        Args:
            text: The input text to be synthesized.
            voice_preset: The voice preset to be used for the synthesis. 
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            _, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)