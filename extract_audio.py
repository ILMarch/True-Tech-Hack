import subprocess
import os


def extract_audio_file(vid, path):
    if not os.path.exists(path):
        os.makedirs(path)
    name = vid[vid.find('/') + 1:vid.find('.')]

    audio_file_path = os.path.join(path, f'{name}.wav')

    command = f"ffmpeg -i {vid} -ab 160k -ac 2 -ar 44100 -vn {audio_file_path}"
    subprocess.call(command, shell=True)
    return audio_file_path


if __name__ == '__main__':
    #extract_audio_file('scenes/002.mp4', path='audio_folder') #'test/vid4.mp4'
    extract_audio_file('test/vid5.mp4', path='audio_folder')
