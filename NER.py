import requests
from dotenv import load_dotenv
import os

# Load environment variables
dotenv_path = '.env.local'# os.path.join(os.path.dirname(__file__), '..', '.env.local')
load_dotenv(dotenv_path)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
      
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
        entities_and_spans = []
        if 'annotations' in ner_results:
            for annotation in ner_results['annotations']:
                entity = annotation.get('mention', '')
                span = annotation.get('span', {})
                if entity and span:
                    entities_and_spans.append({
                        'entity': entity,
                        'span': span
                    })
        
        ner_results['entities_and_spans'] = entities_and_spans
        return ner_results, entities_and_spans

class GPTException(Exception):
    pass

class NERException(Exception):
    pass

def get_gpt_and_query(text, api_key, response_test):
    try:
        gpt_client = GPTClient(api_key)
        ner_client = NERClient()
        
        gpt_response = gpt_client.get_completion(text)
        ner_results, entities_and_spans = ner_client.get_entities(response_test)
        
        return {
            "gpt_response": gpt_response,
            "ner_results": ner_results,
            "entities_and_spans": entities_and_spans
        }
    except (GPTException, NERException) as e:
        return {"error": str(e)}

'''
def get_gpt_and_query(text, api_key, response_test):
    # OpenAI API endpoint
    url = "https://api.openai.com/v1/chat/completions"
    
    # Headers with API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Request payload
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
        # Get response from GPT
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Extract the generated text
        gpt_response = response.json()['choices'][0]['message']['content']
        
        # Send to query_plain
        ner_results = query_plain(response_test)

        # Extract mentions and their spans from NER results
        entities_and_spans = []
        if 'annotations' in ner_results:
            for annotation in ner_results['annotations']:
                entity = annotation.get('mention', '')
                span = annotation.get('span', {})
                if entity and span:
                    entities_and_spans.append({
                        'entity': entity,
                        'span': span
                    })
        
        # Add mentions_and_spans to the response
        ner_results['entities_and_spans'] = entities_and_spans
        
        return {
            "gpt_response": gpt_response,
            "ner_results": ner_results,
            "entities_and_spans": entities_and_spans
        }
        
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    return requests.post(url, json={'text': text}).json()

def jsonprint(data):
    import yaml
    yaml_str = yaml.dump(data)
    print(yaml_str)

'''
    
if __name__ == '__main__':
    text = "Which supplement may slow the progression of Alzheimer's disease?"
    response = """Omega-3 fatty acids are a type of polyunsaturated fat that is essential for human health. They are found in various foods and supplements, with fish oil being a common source. Omega-3s are known for their beneficial effects on cardiovascular health, including reducing triglycerides, lowering blood pressure, and decreasing the risk of heart disease. They also have anti-inflammatory properties that can benefit individuals with conditions like rheumatoid arthritis and may slow the progression of neurodegenerative diseases such as Alzheimer's disease."""
     # print(query_plain(text))
    jsonprint(get_gpt_and_query(text, api_key, response)['gpt_response'])
    print('\n\n')
    jsonprint(get_gpt_and_query(text, api_key, response)['ner_results'])
    print('\n\n')
    jsonprint(get_gpt_and_query(text, api_key, response)['entities_and_spans'])
