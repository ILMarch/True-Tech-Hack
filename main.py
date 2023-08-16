from .vid_to_scenes import split_video_into_scenes
from .scene_to_text import scene_to_text
from .scene_into_frames import split_scene_into_frames
from .translator import translate
import warnings

from tqdm import tqdm
import os
import pandas as pd
import shutil
from .text_to_speech import text_to_speech
from datetime import datetime
import librosa
from pydub import AudioSegment
import subprocess

SCENES_FOLDER = 'scenes'
SCENES_FRAMES_FOLDER = 'scenes_frames'
AUDIO_FOLDER = 'audio_folder'
AUDIO_PATH = 'text_to_speech_folder'

os.environ['TOKENIZERS_PARALLELISM'] = 'False'


def pipeline(path_to_vid, save_path):
    # split video into scenes
    scenes_list = split_video_into_scenes(path_to_vid, dest_path=SCENES_FOLDER)

    # split each scene into frames
    scenes = sorted(os.listdir(SCENES_FOLDER), key=lambda x: int(x[:x.find('.')]))

    folder_names = []
    to_delete = []
    with tqdm(total=len(scenes)) as pbar:
        for i, vid in enumerate(scenes):
            current_video_path = os.path.join(SCENES_FOLDER, vid)
            folder_name = vid[:vid.find('.')]
            folder_names.append(folder_name)
            split_scene_into_frames(
                current_video_path,
                output_folder_path=os.path.join(SCENES_FRAMES_FOLDER, folder_name)
            )
            pbar.update(1)

    scenes_list[:] = [x for i, x in enumerate(scenes_list) if i not in to_delete]
    starts, ends = list(zip(*[(str(i[0]), str(i[1])) for i in scenes_list]))
    captions = []

    # generate and summarize text for each scene
    with tqdm(total=len(folder_names)) as pbar:
        for scene_folder in folder_names:
            captions.append(scene_to_text(os.path.join(SCENES_FRAMES_FOLDER, scene_folder)))
            pbar.update()

    translated_captions = []
    for i, caption in enumerate(captions):
        current_translated = translate(caption)
        translated_captions.append(current_translated)

    out_df = pd.DataFrame(
        {
            'start': starts,
            'end': ends,
            'caption': captions,
            'translated': translated_captions
        }
    )

    out_df = out_df[~out_df['caption'].str.contains('I am a prisoner of the state')]
    out_df = out_df.reset_index(drop=True)

    audio_duration_seconds = 0
    previous_start_seconds = 0

    combined = AudioSegment.empty()

    with tqdm(total=len(out_df)) as pbar:
        for idx, start_time, end_time, caption, ru_text in out_df.itertuples():
            start_datetime = datetime.strptime(start_time, '%H:%M:%S.%f')
            start_seconds = (start_datetime - datetime(1900, 1, 1)).total_seconds()
            end_datetime = datetime.strptime(end_time, '%H:%M:%S.%f')
            end_seconds = (end_datetime - datetime(1900, 1, 1)).total_seconds()

            pause_duration = int((start_seconds - previous_start_seconds - audio_duration_seconds) * 1000)
            previous_start_seconds = start_seconds

            combined = text_to_speech(
                ru_text,
                path=AUDIO_PATH,
                name=f'{idx}',
                pause_duration=pause_duration,
                combined=combined
            )
            current_audio = os.path.join(AUDIO_PATH, f'{idx}.wav')
            audio_duration_seconds = librosa.get_duration(path=current_audio)

            if audio_duration_seconds > (end_seconds - start_seconds):
                tem_path = os.path.join(AUDIO_PATH, 'temp.wav')
                cmd = f'ffmpeg -ss 0 -i {current_audio} -t {int(end_seconds - start_seconds)} {tem_path}'

                subprocess.call(cmd, shell=True)

                os.remove(current_audio)
                os.rename(tem_path, current_audio)

            pbar.update(1)

    combined.export('final_audio.wav', format='wav')

    shutil.rmtree(SCENES_FOLDER)
    shutil.rmtree(SCENES_FRAMES_FOLDER)
    shutil.rmtree(AUDIO_PATH)

    input_video = path_to_vid

    print(input_video)

    input_audio = 'final_audio.wav'

    command = f'ffmpeg -i {input_video} -filter:a "volume=0.5" output_1.mp4'
    command2 = f'ffmpeg -i output_1.mp4 -i input_audio.wav -filter_complex "[0:a][1:a]amerge=inputs=2[a]" -map 0:v -map "[a]" -c:v copy -ac 2 -shortest {save_path}'
    command5 = f'ffmpeg -i {input_audio} -filter:a "volume=4" input_audio.wav'

    subprocess.call(command, shell=True)
    # print('command1 is done')
    subprocess.call(command5, shell=True)
    print('command5 is done')
    subprocess.call(command2, shell=True)
    print('command1 is done')

    os.remove(input_audio)
    os.remove('output_1.mp4')
    os.remove('input_audio.wav')


if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline('../test/vid3.mp4')

