# audio_utils.py

import pyttsx3
import os
from datetime import datetime
from config import OUTPUT_DIR

def generate_audio_file(text: str, filename_prefix: str) -> str | None:
    """
    Generates an audio file from text using pyttsx3 and saves it
    as a reliable .wav file. Returns the path to the saved file.
    """
    try:
        audio_dir = os.path.join(OUTPUT_DIR, "audio")
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        # --- THIS IS THE FIX: Change the file extension to .wav ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(audio_dir, f"{filename_prefix}_{timestamp}.wav") # <-- Changed to .wav
        # ---------------------------------------------------------

        print(f"Generating audio file at: {filepath}")
        
        engine = pyttsx3.init()
        
        # Save the speech to a file. Saving as .wav is much more reliable.
        engine.save_to_file(text, filepath)
        
        # This blocking call is necessary to ensure the file is fully written
        # before our application tries to use it.
        engine.runAndWait()
        
        print("Audio file generated successfully.")
        return filepath

    except Exception as e:
        print(f"Could not generate audio file: {e}")
        return None