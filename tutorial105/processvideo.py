from moviepy.editor import *
import os
cwd=os.getcwd()

clip = VideoFileClip(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial105\output.mp4")
clip.write_images_sequence(
        os.path.join(cwd, "frame%04d.png"), fps=.2 #configure this for controlling frame rate.
    )

