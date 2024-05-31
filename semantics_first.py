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

# recommend similar semantics words (BERT)
def recommend_semantics_BERT(tokens: list[str], top_k=10):
    similar_words_dict = {}

    for token in tokens:
        similar_words_dict[token] = []

        # Tokenize and encode the original word
        original_tokens = BERT_tokenizer(token, return_tensors='pt')
        with torch.no_grad():
            original_outputs = BERT_model(**original_tokens)
        original_embedding = original_outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)

        distances, indices = index_BERT.search(original_embedding, top_k + 1)  # +1 for the token itself
        similar_words = [words_BERT[i] for i in indices[0] if words_BERT[i] != token][:top_k]
        
        similar_words_dict[token] = similar_words

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
def recommend_semantics_sentenceBERT(tokens: list[str], top_k=10):
    similar_words_dict = {}

    for token in tokens:
        similar_words_dict[token] = []

        original_embedding = sentenceBERT_model.encode(token, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

        distances, indices = index_SB.search(original_embedding, top_k + 1)  # +1 for the token itself
        similar_words = [words_SB[i] for i in indices[0] if words_SB[i] != token][:top_k]
        
        similar_words_dict[token] = similar_words

    output = [(word, similar_words_dict[word]) for word in similar_words_dict]
    return output

##############################################