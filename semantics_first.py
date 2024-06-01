import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import random
import faiss
import pickle
import os
from itertools import chain

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
sentenceBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Use wordnet vocab (140K)
# vocab = list(set([word for syn in wn.all_synsets() for word in syn.lemma_names()]))

# Use BERT vocab (30K)
vocab = list(BERT_tokenizer.vocab.keys())

print(f"Vocab size: {len(vocab)}")

######################################### (BERT)

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

if not os.path.exists('./embeddings/vocab_embeddings.pkl'):
    # Calculate embeddings (BERT)
    embeddings_BERT, words = compute_embeddings(vocab, BERT_model, BERT_tokenizer)

    # Save embeddings (BERT)
    with open('./embeddings/vocab_embeddings.pkl', 'wb') as f:
        pickle.dump((embeddings_BERT, words), f)

# Load embeddings (BERT)
with open('./embeddings/vocab_embeddings.pkl', 'rb') as f:
    embeddings_BERT, words_BERT = pickle.load(f)

# Generate FAISS index (BERT)
dimension_BERT = embeddings_BERT.shape[1]
index_BERT = faiss.IndexFlatL2(dimension_BERT)
index_BERT.add(embeddings_BERT)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Recommend similar semantics words (BERT)
def recommend_semantics_BERT(tokens: list[str], top_k=None, threshold=None):
    print(top_k, threshold)
    if top_k is None and threshold is None :
        raise ValueError("Either top_k or threshold should be used.")
    
    similar_words_dict = {}

    for token in tokens:
        similar_words_dict[token] = []

        # Tokenize and encode the original word
        original_tokens = BERT_tokenizer(token, return_tensors='pt')
        with torch.no_grad():
            original_outputs = BERT_model(**original_tokens)
        original_embedding = original_outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)
        distances, indices = index_BERT.search(original_embedding, len(words_BERT))

        if top_k is not None:
            similar_words = [words_BERT[i] for i in indices[0][:top_k + 1] if words_BERT[i] != token]
        else:
            similar_words = [words_BERT[i] for i, dist in zip(indices[0], distances[0]) if dist <= threshold and words_BERT[i] != token]

        # similar_words_dict[token] = similar_words

        # Filter synonyms using WordNet
        word_synsets = wn.synsets(token)
        word_synonyms = set(chain.from_iterable([syn.lemma_names() for syn in word_synsets]))
        filtered_synonyms = [word for word in similar_words if word in word_synonyms]
        similar_words_dict[token] = filtered_synonyms

        # cos_sim_threshold = 0.95
        # similar_embeddings = [embeddings_BERT[i] for i in indices[0] if words_BERT[i] != token]
        # cosine_similarities = [cosine_similarity(original_embedding, emb) for emb in similar_embeddings]
        # filtered_synonyms = [words_BERT[i] for i, sim in zip(indices[0], cosine_similarities) if sim >= cos_sim_threshold]
        # similar_words_dict[token] = filtered_synonyms



    output = [(word, similar_words_dict[word]) for word in similar_words_dict]
    return output

######################################### (sentenceBERT)

def compute_embeddings_sentenceBERT(vocab, model):
    embeddings = []
    words = []

    for word in tqdm(vocab):
        embedding = model.encode(word, convert_to_tensor=True).cpu().numpy()
        embeddings.append(embedding)
        words.append(word)
    
    return np.array(embeddings), words

if not os.path.exists('./embeddings/vocab_embeddings_sentenceBERT.pkl'):
    # Calculate embeddings (sentenceBERT)
    embeddings, words = compute_embeddings_sentenceBERT(vocab, sentenceBERT_model)

    # Save embeddings (sentenceBERT)
    with open('./embeddings/vocab_embeddings_sentenceBERT.pkl', 'wb') as f:
        pickle.dump((embeddings, words), f)

# Load embeddings (sentenceBERT)
with open('./embeddings/vocab_embeddings_sentenceBERT.pkl', 'rb') as f:
    embeddings_SB, words_SB = pickle.load(f)

# Generate FAISS index (sentenceBERT)
dimension_SB = embeddings_SB.shape[1]
index_SB = faiss.IndexFlatL2(dimension_SB)
index_SB.add(embeddings_SB)

# recommend similar semantics words (sentenceBERT)
def recommend_semantics_sentenceBERT(tokens: list[str], top_k=None, threshold=None):
    print(top_k, threshold)
    if top_k is None and threshold is None :
        raise ValueError("Either top_k or threshold should be used.")
    
    similar_words_dict = {}

    for token in tokens:
        similar_words_dict[token] = []
        original_embedding = sentenceBERT_model.encode(token, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = index_SB.search(original_embedding, len(words_SB))

        if top_k is not None:
            similar_words = [words_SB[i] for i in indices[0][:top_k + 1] if words_SB[i] != token]
        else:
            similar_words = [words_SB[i] for i, dist in zip(indices[0], distances[0]) if dist <= threshold and words_SB[i] != token]
        
        # similar_words_dict[token] = similar_words

        # Filter synonyms using WordNet
        word_synsets = wn.synsets(token)
        word_synonyms = set(chain.from_iterable([syn.lemma_names() for syn in word_synsets]))
        filtered_synonyms = [word for word in similar_words if word in word_synonyms]
        similar_words_dict[token] = filtered_synonyms

    output = [(word, similar_words_dict[word]) for word in similar_words_dict]
    return output

##############################################