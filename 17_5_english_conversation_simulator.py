# load xvector containing speaker's voice characteristics from a dataset
import io

import torch
from datasets import load_dataset
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import soundfile as sf
import sounddevice as sd
import  speech_recognition as sr
def pause():
    input('\n[Press Enter to Continue]')

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts")

def text_to_speech(input):
    text = input['response']
    print(text)
    for sentence in text.split('. '):
        speech = pipe(sentence, forward_params={"speaker_embeddings": speaker_embeddings})
        virtual_file = io.BytesIO()
        sf.write(
            virtual_file,
            speech['audio'],
            samplerate=speech['sampling_rate'],
            format='wav'
        )
        virtual_file.seek(0)
        temp_data, temp_sr = sf.read(virtual_file)
        sd.play(temp_data, temp_sr)
        sd.wait()

llm = Ollama(model='llama2')

template = """You are a Staff Engineer working at Google. Currently, you are interviewing a candidate who has applied for the Senior Engineer position. Your task is to first ask the candidate to introduce themselves and then proceed to ask questions related to software engineering. No additional description is needed.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

conversation = ConversationChain(
    prompt=prompt,
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory(),
)

chain = conversation | text_to_speech
chain.invoke(input='Let\'s start!')
pause()

r = sr.Recognizer()

while True:
    # obtain audio from the microphone
    with sr.Microphone(device_index=1) as source:
        print("*Recognition of your speech is active. \n Please say something to the interviewer.*")
        audio = r.listen(source)

    try:
        input_text = r.recognize_google(audio, language="english")
        print(f'me: {input_text}')
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
        continue
    except sr.RequestError as e:
        print("Could not request results from Whisper")
        continue
    if not input_text:
        continue
    chain.invoke(input=input_text)
    pause()