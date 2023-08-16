from transformers import pipeline
import os
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_checkpoint = "Helsinki-NLP/opus-mt-en-ru"
translator = pipeline("translation", model=model_checkpoint, device=DEVICE)


def translate(text):
    return translator(text)[0]['translation_text']


if __name__ == '__main__':
    print(translate('This is my day'))
