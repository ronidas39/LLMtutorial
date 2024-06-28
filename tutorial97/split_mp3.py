from pydub import AudioSegment
import os
cwd=os.getcwd()

def split_audio(input,output,sl):
    audio=AudioSegment.from_file(input)
    total_length=len(audio)
    num_segments=total_length // sl

    for i in range(num_segments):
        start=i*sl
        end=start+sl
        segment=audio[start:end]
        segment_name=f"{output}/segment_{i}.mp3"
        print(segment_name)
        segment.export(segment_name,format="mp3")

    if total_length % sl !=0:
        start=num_segments*sl
        segment=audio[start:]
        segment_name=f"{output}/segment_{num_segments}.mp3"
        segment.export(segment_name,format="mp3")







