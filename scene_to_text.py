from transformers import pipeline  # AutoProcessor, Blip2ForConditionalGeneration
import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
captioner = pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en", device=DEVICE)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device=DEVICE)


def summarize(text):
    return summarizer(text, min_length=5, max_length=15)[0]['summary_text']


def scene_to_text(path_to_scene_frames, translate=False):
    frames = sorted(os.listdir(path_to_scene_frames), key=lambda x: int(x[:x.find('.')]))
    text = ''

    for frame in frames:
        text += captioner(os.path.join(path_to_scene_frames, frame))[0][
            'generated_text']

    return summarize(text)


if __name__ == '__main__':
    scene_to_text('./test/frames3/folder1')
