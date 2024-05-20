import nltk
from nltk.corpus import wordnet as wn
# from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

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
    for token in tokens:
        for word_similar_pron in token[1]:
            for w in original_sentence:
                if token[0] != w[0] and are_meanings_similar(word_similar_pron, w[0], threshold):
                    print("original word: ", token[0], " , ", "replaced word: ", w[0], " , ", "similar word: ", word_similar_pron)

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
# sentenceBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def recommend_semantics_BERT(tokens : list[(str, list[str])], threshold = 0.9):
    original_sentence = " ".join([t[0] for t in tokens])
    original_tokens = BERT_tokenizer(original_sentence, return_tensors='pt', padding=True, truncation=True)
    tokenized_texts = BERT_tokenizer.tokenize(original_sentence)
    with torch.no_grad():
        original_outputs = BERT_model(**original_tokens)
    similar_pron_embeddings = {}
    for token in tokens:
        for word_similar_pron in token[1]:
            word_tokens = BERT_tokenizer(word_similar_pron,return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                word_outputs = BERT_model(**word_tokens)
            word_embedding = word_outputs.last_hidden_state.mean(dim=1).squeeze()
            similar_pron_embeddings[word_similar_pron] = word_embedding

    similarities = {}
    for original_token in tokenized_texts:
        token_id = BERT_tokenizer.convert_tokens_to_ids(original_token)
        token_index = original_tokens['input_ids'][0].tolist().index(token_id)
        token_embedding = original_outputs.last_hidden_state[0][token_index]
        for word, word_embedding in similar_pron_embeddings.items():
            similarity = np.dot(token_embedding.numpy(), word_embedding.numpy()) / (np.linalg.norm(token_embedding.numpy()) * np.linalg.norm(word_embedding.numpy()))
            similarities[(original_token, word)] = similarity
    
    most_similar_word = max(similarities, key=similarities.get)
    return most_similar_word


        






# def recommend_semantics_sentenceBERT(tokens : list[(str, list[str])], threshold = 0.9):
#     original_sentence = [" ".join(t[0]) for t in tokens]

#     embeddings = model.encode(sentences, convert_to_tensor=True)

#     similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

#     for i in range(len(sentences)):
#         for j in range(len(sentences)):
#             print(f"Similarity between \"{sentences[i]}\" and \"{sentences[j]}\": {similarity_matrix[i][j]:.4f}")