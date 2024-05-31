import nltk
from nltk.corpus import wordnet as wn
# from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import random
import faiss
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
# sentenceBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_embeddings(vocab, model, tokenizer):
    embeddings = []
    words = []

    for word in tqdm(vocab):
        tokens = tokenizer(word, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
        words.append(word)
    
    return np.array(embeddings), words

# Calculate embeddings
# vocab = list(BERT_tokenizer.vocab.keys())
# embeddings, words = compute_embeddings(vocab, BERT_model, BERT_tokenizer)

# Save embeddings
# with open('vocab_embeddings.pkl', 'wb') as f:
#     pickle.dump((embeddings, words), f)

# Load embeddings
with open('vocab_embeddings.pkl', 'rb') as f:
    embeddings, words = pickle.load(f)

# Generate FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def recommend_semantics_BERT(tokens: list[str], top_k=10):
    similar_words_dict = {}

    for token in tokens:
        similar_words_dict[token] = []

        # Tokenize and encode the original word
        original_tokens = BERT_tokenizer(token, return_tensors='pt')
        with torch.no_grad():
            original_outputs = BERT_model(**original_tokens)
        original_embedding = original_outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)

        distances, indices = index.search(original_embedding, top_k + 1)  # +1 for the token itself
        similar_words = [words[i] for i in indices[0] if words[i] != token][:top_k]
        
        similar_words_dict[token] = similar_words

    output = [(word, similar_words_dict[word]) for word in similar_words_dict]
    return output





# def recommend_semantics_sentenceBERT(tokens : list[(str, list[str])], threshold = 0.9):
#     original_sentence = [" ".join(t[0]) for t in tokens]

#     embeddings = model.encode(sentences, convert_to_tensor=True)

#     similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

#     for i in range(len(sentences)):
#         for j in range(len(sentences)):
#             print(f"Similarity between \"{sentences[i]}\" and \"{sentences[j]}\": {similarity_matrix[i][j]:.4f}")