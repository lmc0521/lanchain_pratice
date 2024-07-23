lanchain pratice series
===
refrence
--
https://myapollo.com.tw/blog/langchain-hugging-face/

install packages:
--
pip install 
- langchain 
- transformers 
- sentencepiece 
- "datasets[audio]" 
- torchaudio 
- pytorch 
- pydub 
- pyaudio 
- SpeechRecognition 
- openai-whisper

for sound = AudioSegment.from_file(virtual_file, format='wav') ERROR
===
instead with following:
---
temp_data, temp_sr = sf.read(virtual_file)
sd.play(temp_data, temp_sr)
sd.wait()
ref: https://github.com/bastibe/python-soundfile/issues/333