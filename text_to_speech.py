import os
import torch
from ssml_builder.core import Speech
from pydub import AudioSegment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(4)
local_file = 'model.pt'
speaker = 'xenia'  # 'aidar', 'baya', 'kseniya', 'xenia', 'random'
sample_rate = 48000  # 8000, 24000, 48000

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/ru_v3.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)


def text_to_speech(text, path, name, pause_duration, combined):
    if not os.path.exists(path):
        os.makedirs(path)

    speech = Speech()

    speech.prosody(value=text, rate='125%', )
    ssml_text = speech.speak()

    audio_paths = model.save_wav(ssml_text=ssml_text,
                                 speaker=speaker,
                                 sample_rate=sample_rate,
                                 audio_path=os.path.join(path, f'{name}.wav')
                                 )

    pad_ms = pause_duration  # milliseconds of silence needed
    silence = AudioSegment.silent(duration=pad_ms)
    audio = AudioSegment.from_wav(audio_paths)

    padded = silence + audio  # Adding silence after the audio

    combined += padded

    print(audio_paths)

    return combined


if __name__ == '__main__':
    from pydub import AudioSegment

    pad_ms = 1000  # milliseconds of silence needed
    silence = AudioSegment.silent(duration=pad_ms)
    audio = AudioSegment.from_wav('text_to_speech_folder/2.wav')

    padded = silence + audio  # Adding silence after the audio
    padded.export('text_to_speech_folder/padded-file.wav', format='wav')
