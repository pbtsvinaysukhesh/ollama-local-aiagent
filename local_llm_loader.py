# local_llm_loader.py

from langchain_ollama import OllamaLLM as Ollama
from config import OLLAMA_MODEL, TEMPERATURE

def load_local_llm():
    """
    Connects to the local Ollama server to use as the LLM.
    This function assumes the Ollama application is already running.
    """
    print(f"Connecting to Ollama server for model: {OLLAMA_MODEL}")
    
    try:
        # Instantiate the Ollama client, pointing to the desired model
        llm = Ollama(
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
            # Set a long timeout to prevent errors on complex, long-running tasks
            timeout=300 
        )
        
        # Perform a quick, simple invocation to test the connection
        # This will fail immediately if the server is not running or the model is not available.
        llm.invoke("Hi")
        
        print("✅ Successfully connected to Ollama.")
        return llm
        
    except Exception as e:
        # Provide a user-friendly error message if the connection fails
        print(f"❌ Failed to connect to Ollama: {e}")
        print("---")
        print("TROUBLESHOOTING:")
        print("1. Ensure the Ollama application is running on your computer.")
        print(f"2. Ensure you have pulled the model using the command: 'ollama pull {OLLAMA_MODEL}'")
        print("---")
        # Re-raise the exception so the Streamlit app can catch it and display an error
        raise