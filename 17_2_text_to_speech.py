import io
import soundfile as sf
from datasets import load_dataset
import torch
from pydub.playback import play
from transformers import pipeline
from pydub import AudioSegment


embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts")

speech = pipe("Good morning!", forward_params={"speaker_embeddings": speaker_embedding})
virtual_file = io.BytesIO()
sf.write(
    virtual_file,
    speech['audio'],
    samplerate=speech['sampling_rate'],
    format='wav'
)
virtual_file.seek(0)

sound = AudioSegment.from_file(virtual_file, fromat='wav')
# play(sound)
# sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])