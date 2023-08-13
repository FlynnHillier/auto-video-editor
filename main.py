from audio import transcribe_diarized
from machine import clear_gpu
import json

hf_access_token = "" #https://huggingface.co/settings/tokens

audio_file = "media\\audio.wav"
batch_size = 16 #reduce if low gpu mem / increase if high
compute_type = "float32"
whisper_model = "large-v2"
language = "en"

min_speakers : None | int = 2
max_speakers : None | int = 2

clear_gpu()
diarized_transcription = transcribe_diarized(audio_filepath=audio_file,hf_access_token=hf_access_token,whisper_model=whisper_model,batch_size=batch_size,min_speakers=min_speakers,max_speakers=max_speakers,compute_type=compute_type,debug_mode=True)

print(diarized_transcription)
with open("sample.json","w") as f:
    f.write(json.dumps(diarized_transcription,indent=2))