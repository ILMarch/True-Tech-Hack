from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import os


def find_scenes(video_path, threshold=27.0):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    return scene_manager.get_scene_list()


def split_video_into_scenes(video_path, dest_path, threshold=27.0):
    scene_list = find_scenes(video_path, threshold)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    output_file_template = os.path.join(dest_path, '$SCENE_NUMBER.mp4')
    split_video_ffmpeg(video_path, scene_list, show_progress=True, output_file_template=output_file_template)
    return scene_list


if __name__ == '__main__':
    split_video_into_scenes('../test/vid3.mp4', dest_path='./test/frames3')
