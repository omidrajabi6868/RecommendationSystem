import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec
from collections import Counter


def feature_engineering(session_AIDS, session_types):

    for aids, types in zip(session_AIDS, session_types):
        number_of_events = len(aids)
        last_event = aids[-1]
        last_type = types[-1]
        counter_dict = Counter(types)
        clicks_count = counter_dict['clicks']
        carts_count = counter_dict['carts']
        orders_count = counter_dict['orders']

        click_counts_5 = Counter(types[:5])['clicks']
        click_counts_10 = Counter(types[:10])['clicks']
        click_counts_30 = Counter(types[:30])['clicks']

        carts_counts_5 = Counter(types[:5])['carts']
        carts_counts_10 = Counter(types[:10])['carts']
        carts_counts_30 = Counter(types[:30])['carts']

        orders_counts_5 = Counter(types[:5])['orders']
        orders_counts_10 = Counter(types[:10])['orders']
        orders_counts_30 = Counter(types[:30])['orders']

        most_common_5 = Counter(aids[:5]).most_common(1)
        most_common_10 = Counter(aids[:10]).most_common(1)
        most_common_30 = Counter(aids[:30]).most_common(1)
        most_common = Counter(aids).most_common(1)
        return None


def create_word2vec_training_data(session_AIDs, session_types, word2vec, interval_size):
    en_len, de_len = interval_size
    en_aids = []
    en_types = []
    de_aids = []
    de_types = []

    for aids, types in zip(session_AIDs, session_types):
        n = len(aids)
        total_len = 0
        while n > 1:
            if n <= en_len:
                n_x = int(n // 2)
            else:
                n_x = en_len

            if n - n_x <= de_len:
                n_y = n - n_x
            else:
                n_y = de_len

            if n_x < en_len:
                en_aids.append((en_len - n_x) * [-1] + aids[total_len:total_len + n_x])
                en_types.append((en_len - n_x) * [-1] + types[total_len:total_len + n_x])
            else:
                en_aids.append(aids[total_len:total_len + n_x])
                en_types.append(types[total_len:total_len + n_x])
            n -= n_x

            if n_y < de_len:
                de_aids.append(aids[total_len + n_x: total_len + n_x + n_y] + (de_len - n_y) * [-1])
                de_types.append(types[total_len + n_x: total_len + n_x + n_y] + (de_len - n_y) * [-1])
            else:
                de_aids.append(aids[total_len + n_x: total_len + n_x + de_len])
                de_types.append(types[total_len + n_x: total_len + n_x + de_len])
            n -= n_y
            total_len += n_x + n_y

    for i, items in enumerate(en_types):
        items = [np.array(categorize(typ)) for typ in items]
        en_types[i] = items

    for i, items in enumerate(de_types):
        items = [np.array(categorize(typ)) for typ in items]
        de_types[i] = items

    X_de =[]
    Type_de = []
    for aids, types in zip(de_aids, de_types):
        temp_aid = []
        temp_type = []
        for i in range(len(aids)):
            if i == 0:
                temp_aid.append(word2vec.wv.key_to_index[-1])
                temp_type.append(np.array(categorize("")))
            else:
                temp_aid.append(word2vec.wv.key_to_index[aids[i - 1]])
                temp_type.append(types[i - 1])

        X_de.append(temp_aid)
        Type_de.append(temp_type)

    for i, items in enumerate(de_aids):
        items = [word2vec.wv[item] for item in items]
        de_aids[i] = items

    for i, items in enumerate(en_aids):
        items = [word2vec.wv.key_to_index[item] for item in items]
        en_aids[i] = items


    X_en = np.array(en_aids)
    Type_en = np.array(en_types)
    X_de = np.array(X_de)
    Type_x_de = np.array(Type_de)
    Y_de = np.array(de_aids)
    Type_y_de = np.array(de_types)

    return X_en, Type_en, X_de, Type_x_de, Y_de, Type_y_de


# def create_binary_trainig_data(session_AIDs, session_types, interval_size):
#     en_len, de_len = interval_size
#     en_aids = []
#     en_types = []
#     de_aids = []
#     de_types = []
#
#     for aids, types in zip(session_AIDs, session_types):
#         n = len(aids)
#         total_len = 0
#         while n > 1:
#             if n <= en_len:
#                 n_x = int(n // 2)
#             else:
#                 n_x = en_len
#
#             if n - n_x <= de_len:
#                 n_y = n - n_x
#             else:
#                 n_y = de_len
#
#             if n_x < en_len:
#                 en_aids.append((en_len - n_x)*[-1] + aids[total_len:total_len + n_x])
#                 en_types.append((en_len - n_x)*[-1] + types[total_len:total_len + n_x])
#             else:
#                 en_aids.append(aids[total_len:total_len + n_x])
#                 en_types.append(types[total_len:total_len + n_x])
#             n -= n_x
#
#             if n_y < de_len:
#                 de_aids.append(aids[total_len + n_x: total_len + n_x + n_y] + (de_len - n_y)*[-1])
#                 de_types.append(types[total_len + n_x: total_len + n_x + n_y] + (de_len - n_y)*[-1])
#             else:
#                 de_aids.append(aids[total_len + n_x: total_len + n_x + de_len])
#                 de_types.append(types[total_len + n_x: total_len + n_x + de_len])
#             n -= n_y
#             total_len += n_x + n_y
#
#     for i, items in enumerate(en_aids):
#         items = [decimalToBinary(item) for item in items]
#         en_aids[i] = items
#
#     for i, items in enumerate(de_aids):
#         items = [decimalToBinary(item) for item in items]
#         de_aids[i] = items
#
#     for i, items in enumerate(en_types):
#         items = [convert(categorize(typ)) for typ in items]
#         en_types[i] = items
#
#     for i, items in enumerate(de_types):
#         items = [convert(categorize(typ)) for typ in items]
#         de_types[i] = items
#
#     X_en = []
#     X_de = []
#     Y_de = []
#
#     for aids, types in zip(en_aids, en_types):
#         X_en.append([(aid + typ) for aid, typ in zip(aids, types)])
#
#     for aids, types in zip(de_aids, de_types):
#         Y_de.append([(aid + typ) for aid, typ in zip(aids, types)])
#
#     for aids, types in zip(de_aids, de_types):
#         temp = []
#         for i in range(len(aids)):
#             if i == 0:
#                 temp.append(([0]*25 + [1]))
#             else:
#                 temp.append(aids[i-1]+types[i-1])
#
#         X_de.append(temp)
#
#     X_en = np.array(X_en)
#     X_de = np.array(X_de)
#     Y_de = np.array(Y_de)
#
#     return X_en, X_de, Y_de


def categorize(n):
    if n == 'clicks':
        return [1, 0, 0, 0]
    elif n == 'carts':
        return [0, 1, 0, 0]
    elif n == 'orders':
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]


def decimalToBinary(n):
    # converting decimal to binary
    if n == -1:
        str_bin = "".join(["0"]*22)
    else:
        str_bin = bin(n).replace("0b", "")
        if len(str_bin) < 21:
            str_bin = "".join((21 - len(str_bin))*["0"]) + str_bin
        str_bin = '1' + str_bin

    str_bin = convert(str_bin)
    number_bin = [int(s) for s in str_bin]

    return number_bin


def convert(string):
    list1 = []
    list1[:0] = string
    return list1


def prediction(test_file, model, encoder_model, decoder_model, w2v, interval_size, steps=50):
    test_df = pd.read_json(test_file, lines=True)[:1000]
    en_len, de_len = interval_size
    with open('out/prediction.csv', 'w') as f:
        f.write('session_type,labels' + '\n')
        for row in test_df.iterrows():
            session_number = row[1]['session']
            aids = [w2v.wv.key_to_index[event['aid']] for event in row[1]["events"]]
            types = [np.array(categorize(event['type'])) for event in row[1]["events"]]
            n = len(aids)

            if n <= en_len:
                aids = (en_len - n)*[w2v.wv.key_to_index[-1]] + aids
                types = (en_len - n)*[np.array(categorize(-1))] + types
            else:
                aids = aids[-en_len:]
                types = types[-en_len:]

            test_data = np.array(aids)
            test_data = test_data[np.newaxis, ...]
            test_data_type = np.array(types)[np.newaxis, ...]
            target_seq = np.array([w2v.wv.key_to_index[-1]]).reshape(1, 1)
            target_seq_type = np.array([0, 0, 0, 1]).reshape(1, 1, 4)
            state = encoder_model.predict([test_data, test_data_type])
            predicted_output = []
            for i in range(steps):
                yhat, h_f, h_b, c_f, c_b = decoder_model.predict([target_seq, target_seq_type] + state)
                argmaxies = np.argmax(yhat[:, :, -4:], axis=-1)
                argmaxies = to_categorical(argmaxies, 4)
                yhat[:, :, -4:] = argmaxies
                state = [h_f, h_b, c_f, c_b]
                for target, _ in w2v.wv.most_similar(yhat[0, 0, :-4]):
                    predicted_output.append([target, yhat[0, 0, -4:]])
                target_seq = np.array(w2v.wv.key_to_index[w2v.wv.most_similar(yhat[0, 0, :-4])[0][0]]).reshape(1, 1)
                target_seq_type = yhat[:, :, -4:]

            clicks = []
            carts = []
            orders = []
            for aid_num, ty in predicted_output:
                if ty[-1] == 1:
                    continue
                if ty[-2] == 1:
                    orders.append(aid_num)
                if ty[-3] == 1:
                    carts.append(aid_num)
                if ty[-4] == 1:
                    clicks.append(aid_num)

            def rank_list(the_list):
                element_freq = {}
                for element in the_list:
                    if element in element_freq:
                        element_freq[element] += 1
                    else:
                        element_freq[element] = 1
                # sort the dictionary by values in descending order
                sorted_elements = sorted(element_freq, key=element_freq.get, reverse=True)

                return sorted_elements[:20]

            f.write(str(session_number) + '_clicks,' + " ".join(str(click) for click in rank_list(clicks)) + '\n')
            f.write(str(session_number) + '_carts,' + " ".join(str(cart) for cart in rank_list(carts)) + '\n')
            f.write(str(session_number) + '_orders,' + " ".join(str(order) for order in rank_list(orders)) + '\n')