from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import openai

openai.api_key = "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"

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512*2
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
    # gpt-3.5-turbo, gpt-4
    model_vesion = 'gpt-4-turbo' #'gpt-3.5-turbo'
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name=model_vesion, max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="ChatMaPS")

index = construct_index("library")

iface.launch(share=True)