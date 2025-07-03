# -*- coding: utf-8 -*-
"""

@author: r_jairam
"""

import os
from pydub import AudioSegment

def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0

def calculate_class_durations(main_folder):
    durations = {}
    for class_folder in os.listdir(main_folder):
        class_path = os.path.join(main_folder, class_folder)
        if os.path.isdir(class_path):
            total_duration = 0.0
            for audio_file in os.listdir(class_path):
                if audio_file.endswith('.wav'):
                    total_duration += get_audio_duration(os.path.join(class_path, audio_file))
            durations[class_folder] = total_duration / 3600.0
    return durations

if __name__ == "__main__":
    folder = 'Audio_directory'
    durations = calculate_class_durations(folder)
    for k, v in durations.items():
        print(f"{k}: {v:.2f} hours")
