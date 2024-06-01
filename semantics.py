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

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
sentenceBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Use wordnet vocab (140K)
# vocab = list(set([word for syn in wn.all_synsets() for word in syn.lemma_names()]))

# Use BERT vocab (30K)
vocab = list(BERT_tokenizer.vocab.keys())

print(f"Vocab size: {len(vocab)}")

##################################################################### (WordNet)
brown_ic = wn.ic(nltk.corpus.brown, False, 0.0)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def are_meanings_similar(word1, word2, threshold):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    max_similarity = 0.0

    for synset1 in synsets1:
        for synset2 in synsets2:
            # similarity = synset_sim.res_similarity(synset_original, brown_ic)
            similarity = synset1.wup_similarity(synset2)
            if similarity and similarity > max_similarity:
                max_similarity = similarity

    return max_similarity > threshold

def recommend_semantics_wordnet(tokens : list[(str, list[str])], threshold = 0.8):
    original_sentence = nltk.pos_tag([t[0] for t in tokens])
    new_sentences = []
    for token in tokens:
        for word_similar_pron in token[1]:
            for w in original_sentence:
                if token[0] != w[0] and are_meanings_similar(word_similar_pron, w[0], threshold):
                    new_sentence = [word_similar_pron if element[0] == w[0] else element[0] for element in original_sentence]
                    new_sentences.append(new_sentence)

    return new_sentences

############################################################################ (BERT)

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

# Recommend similar semantics words, given similar pronunciation words (BERT)
def recommend_semantics_BERT(tokens: list[(str, list[str])], threshold=0.9):
    embedding_dict = {word: embeddings_BERT[words_BERT.index(word)] for word in words_BERT}
    
    original_sentence = nltk.pos_tag([t[0] for t in tokens])
    new_sentences = []
    for token in tokens:
        for word_similar_pron in token[1]:
            for w in original_sentence:
                if token[0] != w[0]:
                    if word_similar_pron in embedding_dict and w[0] in embedding_dict:
                        similar_pron_embedding = embedding_dict[word_similar_pron]
                        other_token_embedding = embedding_dict[w[0]]
                        similarity = np.dot(similar_pron_embedding, other_token_embedding) / (np.linalg.norm(similar_pron_embedding) * np.linalg.norm(other_token_embedding))
                        if similarity >= threshold:
                            new_sentence = [word_similar_pron if element[0] == w[0] else element[0] for element in original_sentence]
                            new_sentences.append(new_sentence)

    return new_sentences

########################################################### (sentenceBERT)

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

def recommend_semantics_sentenceBERT(tokens: list[(str, list[str])], threshold=0.9):
    embedding_dict = {word: embeddings_SB[words_SB.index(word)] for word in words_SB}
    
    original_sentence = nltk.pos_tag([t[0] for t in tokens])
    new_sentences = []

    for token in tokens:
        for word_similar_pron in token[1]:
            for w in original_sentence:
                if token[0] != w[0]:
                    if word_similar_pron in embedding_dict and w[0] in embedding_dict:
                        similar_pron_embedding = embedding_dict[word_similar_pron]
                        other_token_embedding = embedding_dict[w[0]]
                        similarity = np.dot(similar_pron_embedding, other_token_embedding) / (np.linalg.norm(similar_pron_embedding) * np.linalg.norm(other_token_embedding))
                        if similarity >= threshold:
                            new_sentence = [word_similar_pron if element[0] == w[0] else element[0] for element in original_sentence]
                            new_sentences.append(new_sentence)

    return new_sentences

