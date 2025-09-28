# config.py

import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# --- OLLAMA MODEL CONFIGURATION ---
# This is the name of the model you have pulled with "ollama pull"
# You can change this to "mistral", "llama3", etc. if you pull them.
OLLAMA_MODEL = "gemma3:4b"

# --- LLM GENERATION PARAMETERS ---
# This controls the "creativity" of the AI. Higher is more creative.
TEMPERATURE = 0.6