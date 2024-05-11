import nltk
from nltk.corpus import wordnet as wn

brown_ic = wn.ic(nltk.corpus.brown, False, 0.0)

# I have forgotten that tomato is rotten.

words = [('I', ["I"]),('have', ['have']),('forgotten', ['rotten', "got", "garden"]), ('.', ['.'])]

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

def recommend_semantics_wordnet(mapped_words):

    output = ""
    original_sentence = nltk.pos_tag([wp[0] for wp in mapped_words])
    for wp1 in mapped_words:
        for word_similar_pron in wp1[1]:
            for syn_sim in wn.synsets(word_similar_pron):
                for w in original_sentence:
                    for syn_original in wn.synsets(w[0]):
                        if syn_original.pos() == get_wordnet_pos(w[1]) and syn_original.pos() == syn_sim.pos():
                            similarity = syn_sim.res_similarity(syn_original, brown_ic)
                            if similarity > 3:
                                print("original word: ", syn_original, "similar word: ", syn_sim, "similarity: ", similarity)
                                print("sim: ", wp1[1])
                                print("original: ", w[0])

# recommend_semantics_wordnet(words)
a = wn.synsets("i")
s1 = wn.synset('dog.n.01').pos()
print(a)
# s2 = wn.synset("pencil.n.01")
# print(s1.res_similarity(s2,brown_ic))