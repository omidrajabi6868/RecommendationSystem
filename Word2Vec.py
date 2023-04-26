GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
import sys
from multiprocessing import cpu_count
import gc
from tqdm import tqdm
import datetime

import polars as pl
import pandas as pd
import numpy as np
from collections import Counter

from gensim.test.utils import common_texts
from gensim.models import Word2Vec


class CFG:
    debug = False
    embed_dim = 32
    tz = datetime.timezone.utc
    ts_zero = datetime.datetime(1970, 1, 1, tzinfo=tz)
    contentType_mapper = pd.Series(["clicks", "carts", "orders"], index=[0, 1, 2])
    target_weight = (0.1, 0.3, 0.6)

    w2v_type = {
        "CBOW": 0,
        "SkipGram": 1
    }


sentences_df = pl.concat([pl.read_parquet('out/train.parquet'), pl.read_parquet('out/test.parquet')])
sentences_df = sentences_df[["session", "aid"]].groupby("session").agg([pl.col("aid")])
sentences_df = sentences_df["aid"].to_list()[:1000] if CFG.debug else sentences_df["aid"].to_list()
w2vec = Word2Vec(sentences=sentences_df, sg=CFG.w2v_type["CBOW"], vector_size=CFG.embed_dim, window=2,
                 min_count=1, compute_loss=True, seed=42, workers=cpu_count())
w2vec.save("Models/word2vec.model")

new_word = str(-1)  # convert the integer to a string
new_vector = np.zeros(shape=(32,))

# Add the new word to the vocabulary
w2vec.wv.add(new_word, new_vector)

# Train the model with the new word
w2vec.train([new_word], total_examples=1, epochs=1)

w2vec.save("Models/word2vec.model")
