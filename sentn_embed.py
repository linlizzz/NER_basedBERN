import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the spaCy model (make sure to install it first)
nlp = spacy.load('en_core_web_md')  # Use 'en_core_web_md' or 'en_core_web_lg' for better accuracy

# Convert the query and sentences to vectors
query = "User's query"
sentences = ["Sentence one.", "Sentence two.", "Sentence three."]

query_vector = nlp(query).vector
sentence_vectors = [nlp(sentence).vector for sentence in sentences]

# Load pre-trained BERT model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

query_vector = embed(query)
sentence_vectors = [embed(sentence) for sentence in sentences]

query_vector = query_vector.reshape(1, -1)
sentence_vectors = np.array(sentence_vectors)
similarities = cosine_similarity(query_vector, sentence_vectors).flatten()

# Rank the sentences by similarity
most_similar_indices = np.argsort(-similarities)
most_relevant_sentences = [sentences[i] for i in most_similar_indices]

print("Most relevant sentences to the query:")
for idx, sentence in enumerate(most_relevant_sentences):
    print(f"{idx + 1}: {sentence} (Similarity: {similarities[most_similar_indices[idx]]})")
