import requests
from dotenv import load_dotenv
import os

from typing import List, Optional
from neo4j import GraphDatabase

import openai
import numpy as np
import pandas as pd

from AI_Agent import AI_respnse, match_KG_nodes, visualization


# Load KG nodes embedding
kg_nodes_embedding = pd.read_parquet("ADInt_CUI_embeddings.parquet")
# Load environment variables
dotenv_path = '.env.local'# os.path.join(os.path.dirname(__file__), '..', '.env.local')
load_dotenv(dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")
neo4j_url = os.getenv("NEO4J_URL")
username = os.getenv("USER_NAME")
password = os.getenv("PASSWORD")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Add this check after loading environment variables
if not neo4j_url:
    raise ValueError("NEO4J_URL not found in environment variables")

# Initialize OpenAI client with the API key
openai.api_key = api_key
      
class GPTClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_completion(self, text):
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ]
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise GPTException(f"GPT API request failed: {str(e)}")

class NERClient:
    def __init__(self, url="http://bern2.korea.ac.kr/plain"):
        self.url = url
    
    def get_entities(self, text):
        try:
            ner_results = requests.post(self.url, json={'text': text}).json()
            return self._process_ner_results(ner_results)
        except requests.exceptions.RequestException as e:
            raise NERException(f"NER API request failed: {str(e)}")
    
    def _process_ner_results(self, ner_results):
        entities = []
        entities_and_spans = []
        if 'annotations' in ner_results:
            # Get all mentions
            entities = [annotation.get('mention', '') for annotation in ner_results['annotations']]
            # Get entities and spans as before
            entities_and_spans = [
                {
                    'entity': annotation.get('mention', ''),
                    'span': annotation.get('span', {})
                }
                for annotation in ner_results['annotations']
            ]
        
        return entities, entities_and_spans

class GPTException(Exception):
    pass

class NERException(Exception):
    pass

def bernner_respnse(query_text, response_text, api_key=api_key):
    try:
        gpt_client = GPTClient(api_key)
        ner_client = NERClient()
        
        gpt_response_original = gpt_client.get_completion(query_text)

        ner_query, query_entities_and_spans = ner_client.get_entities(query_text)
        ner_response_original, response_entities_and_spans_original = ner_client.get_entities(response_text)
        
        return [gpt_response_original, ner_query, query_entities_and_spans, ner_response_original, response_entities_and_spans_original]
    
    except (GPTException, NERException) as e:
        return {"error": str(e)}
    


def agent(kg_nodes_embedding, user_input, option = "combined"):

    # print("keywords from gpt:")
    qa_response_prompt, keywords_list_answer_prompt, keywords_list_question_prompt = AI_respnse(user_input)
    # print(qa_response_prompt)
    # print(keywords_list_answer_prompt)
    # print(keywords_list_question_prompt)

    # print("keywords from bern:")
    gpt_response_original, ner_query, _, ner_response_original, _ = bernner_respnse(user_input, response_text=qa_response_prompt)
    # print(gpt_response_original)
    # print(ner_query)
    # print(ner_response_original)

    if option == "bern":
        keywords_list_answer = ner_response_original
        keywords_list_question = ner_query
    elif option == "gpt":
        keywords_list_answer = keywords_list_answer_prompt
        keywords_list_question = keywords_list_question_prompt
    elif option == "combined":
        def combine_unique_lists(list1, list2):
            combined = set([item.lower() for item in list1] + [item.lower() for item in list2])
            return list(combined)
        keywords_list_answer = combine_unique_lists(ner_response_original, keywords_list_answer_prompt)
        keywords_list_question = combine_unique_lists(ner_query, keywords_list_question_prompt)
    else:
        print("The option has not been defined!")

    print("Extracted entities from query: ")
    print(keywords_list_question)
    print("Extracted entities from response: ")
    print(keywords_list_answer)

    keywords_list = combine_unique_lists(keywords_list_answer, keywords_list_question)
    # Similarity Search
    nodes_list_answer = match_KG_nodes(keywords_list_answer, kg_nodes_embedding)
    nodes_list_question = match_KG_nodes(keywords_list_question, kg_nodes_embedding)
    print("Matched entities from query: ")
    print(nodes_list_answer)
    print("Matched entities from response: ")
    print(nodes_list_question)
    
    nodes_list = match_KG_nodes(keywords_list, kg_nodes_embedding)
    vis_res = visualization(nodes_list)
    # vis_res = visualization_neo4j(nodes_list, neo4j_url, username, password)
    '''
    vis_res: List[Dict[str, Union[List[Dict], List[Dict]]]]
        each dictionary has two main keys:

    1. 'nodes': Contains a list of node dictionaries with properties:
            Node_ID: int
            CUI: str
            Name: str
            Label: str
    2. 'edges': Contains a list of edge dictionaries with properties:
            Relation_ID: int
            Source: int
            Target: int
            Type: Optional[str]
            PubMed_ID: str

'''
    print("subgraph results:")
    print(vis_res)


if __name__ == '__main__':
    user_input = input("Please enter your healthcare-related question (type E to use the example): ")
    if user_input == "E":
        user_input = "Can Ginkgo biloba and vitamin E prevent Alzheimer's Disease?"
    print("Query: ", user_input)
    
    agent(kg_nodes_embedding, user_input, option = "combined")
   