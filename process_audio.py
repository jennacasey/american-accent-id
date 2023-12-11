import os
from transformers import pipeline

pipe = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")

audio_directory = "/Users/jennacasey/Downloads/new_wavs"

for audio_file in os.listdir(audio_directory):
    if audio_file.endswith(".wav"):
        audio_path = os.path.join(audio_directory, audio_file)

        output = pipe(audio_path, chunk_length_s=10, stride_length_s=(4, 2))

        output_file = os.path.splitext(audio_file)[0] + "_output.txt"

        with open(output_file, "w") as f:
            f.write(str(output))

        print(f"Processed {audio_file} and saved output to {output_file}")
