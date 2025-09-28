# model_manager.py

import os
from langchain_community.llms import LlamaCpp
from config import (
    POWERFUL_MODEL_PATH, POWERFUL_MODEL_NAME,
    FAST_MODEL_PATH, FAST_MODEL_NAME,
    MAX_NEW_TOKENS, CPU_CONFIG, HYBRID_CONFIG, GPU_CONFIG
)

class ModelManager:
    def __init__(self):
        self.llm_powerful = None
        self.llm_fast = None

    def load_models(self, powerful_model_mode: str = "hybrid"):
        """Loads both the powerful and fast LLMs into memory."""
        
        # --- LOAD THE POWERFUL WRITER MODEL ---
        if not os.path.exists(POWERFUL_MODEL_PATH):
            raise FileNotFoundError(f"Powerful LLM model not found at {POWERFUL_MODEL_PATH}.")
        
        print(f"Loading POWERFUL model '{POWERFUL_MODEL_NAME}' in '{powerful_model_mode.upper()}' mode...")
        config_map = {"cpu": CPU_CONFIG, "hybrid": HYBRID_CONFIG, "gpu": GPU_CONFIG}
        powerful_config = config_map.get(powerful_model_mode.lower())
        if not powerful_config: raise ValueError("Invalid mode for powerful model.")

        self.llm_powerful = LlamaCpp(
            model_path=POWERFUL_MODEL_PATH,
            max_tokens=MAX_NEW_TOKENS,
            n_batch=512,
            n_ctx=4096,
            verbose=True,
            **powerful_config
        )
        print("✅ Powerful LLM loaded successfully.")

        # --- LOAD THE FAST PLANNER MODEL ---
        if not os.path.exists(FAST_MODEL_PATH):
            raise FileNotFoundError(f"Fast LLM model not found at {FAST_MODEL_PATH}.")
            
        print(f"Loading FAST model '{FAST_MODEL_NAME}'...")
        # The fast model is small, so we can use a safe, lightweight config.
        # We offload a few layers to the GPU if possible, but keep it minimal.
        fast_config = {"n_gpu_layers": 5}

        self.llm_fast = LlamaCpp(
            model_path=FAST_MODEL_PATH,
            max_tokens=1024, # Outlines and summaries are shorter
            n_batch=512,
            n_ctx=4096,
            verbose=False, # Keep logs cleaner
            **fast_config
        )
        print("✅ Fast LLM loaded successfully.")