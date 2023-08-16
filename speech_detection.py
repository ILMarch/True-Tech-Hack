import whisper
import torch
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("medium", device=device)


def detect_speech(path_to_audio, threshold=0.64, len_threshold=1, language='en'):
    result = model.transcribe(path_to_audio, language=language)

    if 'segments' in result.keys() and len(result['segments']) > 0:
        mean_prob = np.array([result['segments'][i]['no_speech_prob'] for i in range(len(result['segments']))]).mean()

        return mean_prob <= threshold and len(result['text']) > len_threshold


if __name__ == '__main__':
    detect_speech('audio_folder/vid5.wav')
