import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm


def create_trainig_data(session_AIDs, session_types, interval_size):
    splited_aids = []
    splited_types = []
    for aids, types in zip(session_AIDs, session_types):
        n = len(aids)
        i = 0
        while n > 1:
            if n < interval_size:
                splited_aids.append((interval_size - n)*[-1] + aids[interval_size*i:interval_size*i + n])
                splited_types.append((interval_size - n)*[-1] + types[interval_size*i:interval_size*i + n])
            else:
                splited_aids.append(aids[interval_size*i: interval_size*(i+1)])
                splited_types.append(types[interval_size*i: interval_size*(i+1)])
            n -= interval_size
            i += 1

    for i, items in enumerate(splited_aids):
        items = [decimalToBinary(item) for item in items]
        splited_aids[i] = items

    for i, items in enumerate(splited_types):
        items = [convert(categorize(type)) for type in items]
        splited_types[i] = items

    X = []
    Y = []
    for aids, types in zip(splited_aids, splited_types):
        Y.append([(aid + type) for aid, type in zip(aids, types)])

    for aids, types in zip(splited_aids, splited_types):
        temp = []
        for i in range(len(aids)):
            if i == 0:
                temp.append(([0]*25 + [1]))
            else:
                temp.append(aids[i-1]+types[i-1])

        X.append(temp)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


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


def prediction(test_file, model, interval_size=20, steps=100):
    test_df = pd.read_json(test_file, lines=True)

    with open('out/prediction.csv', 'w') as f:
        f.write('session_type,labels' + '\n')
        for row in test_df.iterrows():
            session_number = row[1]['session']
            aids = [decimalToBinary(event['aid']) for event in row[1]["events"]]
            types = [convert(categorize(event['type'])) for event in row[1]["events"]]
            n = len(aids)
            if n < interval_size:
                aids = (interval_size - n)*[decimalToBinary(-1)] + aids
                types = (interval_size - n)*[convert(categorize(-1))] + types
            else:
                aids = aids[-interval_size:]
                types = types[-interval_size:]

            test_data = [aid + type for aid, type in zip(aids, types)]
            test_data = np.array(test_data)
            test_data = test_data[np.newaxis, ...]
            predicted_output = []
            for i in range(steps):
                pre = model.predict(test_data)
                pre = np.array(pre > .5, dtype='int32')
                predicted_output.append(list(pre[0, -1]))
                test_data = test_data[0, 1:, :]
                test_data = np.concatenate([test_data, pre[:, -1, :]], axis=0)
                test_data = test_data[np.newaxis, ...]

            clicks = []
            carts = []
            orders = []
            for item in predicted_output:
                if item[-1] == 1:
                    continue

                binary_num = item[1:-4]
                aid_num = int("".join([str(num) for num in binary_num]), 2)
                if item[-2] == 1:
                    if aid_num not in orders:
                        orders.append(aid_num)
                if item[-3] == 1:
                    if aid_num not in carts:
                        carts.append(aid_num)
                if item[-4] == 1:
                    if aid_num not in clicks:
                        clicks.append(aid_num)

            f.write(str(session_number) + '_clicks,' + " ".join(str(click) for click in clicks[:20]) + '\n')
            f.write(str(session_number) + '_carts,' + " ".join(str(cart) for cart in carts[:20]) + '\n')
            f.write(str(session_number) + '_orders,' + " ".join(str(order) for order in orders[:20]) + '\n')