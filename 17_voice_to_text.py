import speech_recognition
print(speech_recognition.Microphone.list_microphone_names())
mic=speech_recognition.Microphone(device_index=1)

# # obtain audio from the microphone
r = speech_recognition.Recognizer()
# with speech_recognition.AudioFile("./speech.wav") as source:
#     audio=r.record(source)

with mic as source:
    print("Say something!")
    audio = r.listen(source)

try:
    text = r.recognize_google(audio, language="english")
    print("Whisper thinks you said " + text)
except speech_recognition.UnknownValueError:
    print("Whisper could not understand audio")
except speech_recognition.RequestError as e:
    print("Could not request results from Whisper")