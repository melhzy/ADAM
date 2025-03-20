import os
import openai
import gradio as gr
from llama_index import (
    SimpleDirectoryReader, 
    GPTVectorStoreIndex, 
    LLMPredictor,
    PromptHelper
)
from langchain.chat_models import ChatOpenAI

class AIRAApp:
    CONTEXT_WINDOW = 4096
    NUM_OUTPUT = 512
    CHUNK_OVERLAP_RATIO = 0.1
    CHUNK_SIZE_LIMIT = 600
    MODEL_NAME = "gpt-3.5-turbo"

    def __init__(self, api_key, directory_path):
        """
        Initialize AIRA app with API key and data directory path.

        Parameters:
            api_key (str): API key for OpenAI.
            directory_path (str): Path to the data directory.
        """
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        self.index = self.construct_index(directory_path)

    def construct_index(self, directory_path):
        """
        Constructs an index using specified directory path and predefined configurations.

        Parameters:
            directory_path (str): Path to the directory containing the data.
        
        Returns:
            GPTVectorStoreIndex: Constructed index.
        """
        prompt_helper = PromptHelper(
            context_window=self.CONTEXT_WINDOW, 
            num_output=self.NUM_OUTPUT, 
            chunk_overlap_ratio=self.CHUNK_OVERLAP_RATIO,
            chunk_size_limit=self.CHUNK_SIZE_LIMIT
        ) 
        
        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(
                temperature=0.7, 
                model_name=self.MODEL_NAME, 
                max_tokens=self.NUM_OUTPUT
            )
        )
        documents = SimpleDirectoryReader(directory_path).load_data()
        index = GPTVectorStoreIndex(
            documents, 
            llm_predictor=llm_predictor, 
            prompt_helper=prompt_helper
        )
        index.storage_context.persist()
        
        return index

    def query_index(self, input_text):
        """
        Queries the constructed index with the input text and returns a response.
        
        Parameters:
            input_text (str): Text input for the query.
            
        Returns:
            str: Response from the query.
        """
        query_engine = self.index.as_query_engine()
        response = query_engine.query(input_text)
        return response.response

    def launch_interface(self, inbrowser=False):
        """
        Launches the Gradio interface.
        
        Parameters:
            inbrowser (bool): Whether to open the Gradio interface in the browser.
        """
        iface = gr.Interface(
            fn=self.query_index,
            inputs=gr.components.Textbox(lines=7, label="Enter your text"),
            outputs=gr.components.Textbox(lines=7, label="Output"),
            title="AIRA - Artificial Intelligence Research Assistant"
        )
        iface.launch(inbrowser=inbrowser)


# Usage:
# api_key = os.getenv('OPENAI_API_KEY')  # Getting API key from environment variables
# app = AIRAApp(api_key, "diabetes")
# app.launch_interface()