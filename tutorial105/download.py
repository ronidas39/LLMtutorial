
from pytubefix import YouTube
from pytubefix.cli import on_progress
url = "https://www.youtube.com/watch?v=j7_L3w4vOJk&t=25s"
yt = YouTube(url)
print(yt.title)
ys = yt.streams.get_highest_resolution()
ys.download(filename="output.mp4")
