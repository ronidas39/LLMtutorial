from openai import OpenAI
import os,glob,io
client=OpenAI()
cwd=os.getcwd()
def generate_transcript():
    files=glob.glob(cwd+"\*.mp3")
    print(files)
    with io.open("transcript.txt","w",encoding="utf-8") as f1:
        for file in files:
            name=file.split("\\")[-1]
            if name != "sample.mp3":
                audio_file=open(file,"rb")
                transcription=client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                print(transcription.text)
                f1.write(transcription.text+"\n")

