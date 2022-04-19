import pickle
import os
import gensim
from urllib.request import urlretrieve, urlopen
import pandas as pd
import numpy as np
from args import parse_args

def gen_numerize(tp, show2id):
    sid = tp['item'].apply(lambda x: show2id[x])
    return sid

def item_genre_emb_mean(i):
    total.append(np.mean(gen[gen['item'] == i].emb))

args = parse_args(mode="train")

pro_dir = os.path.join(args.data_dir, 'pro_sg')
with open(os.path.join(pro_dir, "profile2id.pkl"), 'rb') as f:
    profile2id = pickle.load(f)
with open(os.path.join(pro_dir, "show2id.pkl"), 'rb') as f:
    show2id = pickle.load(f)

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
gen = pd.read_csv(os.path.join(args.data_dir, "genres.tsv", delimiter='\t'))
emb_list = []
total = []


gen['item'] = gen_numerize(gen, show2id)
gen_emb = pd.DataFrame(gen.genre.value_counts().index.values, columns=['genre'])

for x in gen_emb.genre:
    if x == 'Sci-Fi':
        emb_list.append(word2vec_model['science_fiction'])
    elif x == 'Film-Noir':
        emb_list.append(word2vec_model['Film_Noir'])
    else:
        emb_list.append(word2vec_model[x])

x = pd.concat([gen_emb, pd.DataFrame(emb_list)], axis=1)
a = x.set_index('genre', drop=True)

gen2emb = dict((x, a.loc[x].values) for (i, x) in enumerate(a.index))
gen['emb'] = gen['genre'].apply(lambda x: gen2emb[x])


item_genre_emb_idx = pd.DataFrame(list(i for i in range(0, max(gen.item)+1)), columns=['item'])
item_genre_emb_idx.item.apply(lambda x: item_genre_emb_mean(x))
item_genre_emb = pd.DataFrame(total)
item_genre_emb = item_genre_emb.T
item_genre_emb.to_csv('item_genre_eb.csv')
