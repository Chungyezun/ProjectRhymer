import nltk
from nltk.corpus import wordnet as wn

brown_ic = wn.ic(nltk.corpus.brown, False, 0.0)

# words = [('I', ["I"]),('have', ['have']),('forgotten', ['rotten', "got", "garden"]), ('.', ['.'])]

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
    # output = ""
    original_sentence = nltk.pos_tag([t[0] for t in tokens])
    for token in tokens:
        for word_similar_pron in token[1]:
            for w in original_sentence:
                if token[0] != w[0] and are_meanings_similar(word_similar_pron, w[0], threshold):
                    print("original word: ", token[0], " , ", "replaced word: ", w[0], " , ", "similar word: ", word_similar_pron)