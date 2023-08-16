import ffmpeg
import os

def split_scene_into_frames(path_to_scene, output_folder_path=None):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    try:
        (ffmpeg.input(path_to_scene)
         .filter('fps', fps=0.5)
         .output(f'{output_folder_path}/%d.png',
                 video_bitrate='5000k',
                 s='224x224',
                 sws_flags='bilinear',
                 start_number=0)
         .run(capture_stdout=True, capture_stderr=True))
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))



if __name__ == '__main__':
    split_scene_into_frames('./test/frames3/vid3-Scene-001.mp4', './test/frames3/folder1')