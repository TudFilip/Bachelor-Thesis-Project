import subprocess
import os


def convert_to_mp4(input_file, output_file):
    cmd = ['ffmpeg', '-i', input_file, '-an', '-codec:v', 'copy', output_file]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("The input file does not have a video stream.")


def convert_files():
    input_folder_path = '../Movies'
    output_folder_path = '../Movies_MP4'
    video_files = [f for f in os.listdir(input_folder_path) if f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.mkv')]
    for video_file in video_files:
        input_file = os.path.join(input_folder_path, video_file)

        if video_file.endswith('.mp4'):
            output_file = os.path.join(output_folder_path, video_file)
            convert_to_mp4(input_file, output_file)
            continue

        base_video_file_name, extension = os.path.splitext(video_file)
        new_video_file_name = base_video_file_name + '.mp4'
        output_file = os.path.join(output_folder_path, new_video_file_name)

        convert_to_mp4(input_file, output_file)


def convert_movie_to_mp4(movie_path):
    output_folder_path = '../Movies_MP4'
    if movie_path.endswith('.mp4') or movie_path.endswith('.avi') or movie_path.endswith('.mkv'):
        base_movie_path = os.path.splitext(movie_path)[0]
        movie_name = os.path.basename(base_movie_path)
        new_movie_name = movie_name + '.mp4'
        output_file = os.path.join(output_folder_path, new_movie_name)
        convert_to_mp4(movie_path, output_file)


