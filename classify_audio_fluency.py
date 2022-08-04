import numpy as np
import os
import pandas as pd
import sklearn
import sklearn.ensemble as se
import sklearn.model_selection as sms
import tensorflow as tf

import beta_audio

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


tf.linalg.inv(tf.random.normal([2, 2]))


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(
    root_dir, 'BetaData', 'coach', 'ml', 'fluency')


# Feature extraction

audio_subdirectories = os.listdir(data_root_dir)
audio_subdirectories = sorted(audio_subdirectories)
print(f'Audio subdirs: {audio_subdirectories}')

n_mfccs, n_rmse, n_sf, n_zcr = 20, 1, 1, 1
feature_dims = [n_mfccs, n_rmse, n_sf, n_zcr]
feature_gate = [1, 1, 1, 1]
features, labels, files = beta_audio.parse_audio_files(
    data_root_dir, audio_subdirectories, feature_dims, feature_gate)
print()


# Data exploration

label_classes = np.array(audio_subdirectories)
n_features = features.shape[-1]
n_classes = len(label_classes)

print(f'Data splitting summary')
print(f'Dataset total length:  {len(labels)}')
x_raw, x_test, y_raw, y_test, f_raw, f_test = sms.train_test_split(
    features, labels, files, test_size=0.1, stratify=labels)
x_train, x_val, y_train, y_val, f_train, f_val = sms.train_test_split(
    x_raw, y_raw, f_raw, train_size=0.85, stratify=y_raw)
print(f'Training set length:   {len(y_train)}')
print(f'Validation set length: {len(y_val)}')
print(f'Test set length:       {len(y_test)}')
print()

print(pd.DataFrame(
    {'Label': label_classes,
     'Training': [y_train.tolist().count(i) for i in range(len(label_classes))],
     'Validation': [y_val.tolist().count(i) for i in range(len(label_classes))],
     'Test': [y_test.tolist().count(i) for i in range(len(label_classes))]
     }))
print()


# MLP
print('MLP')

mlps = []

best_mlp_loss, best_mlp_acc = 0, 0
best_shot_loss, best_shot_acc = 0, 0

i_count, j_count = 100, 10

for i in range(i_count):

    x_raw, x_test, y_raw, y_test, f_raw, f_test = sms.train_test_split(
        features, labels, files, test_size=0.1,
        random_state=i, stratify=labels)
    x_train, x_val, y_train, y_val, f_train, f_val = sms.train_test_split(
        x_raw, y_raw, f_raw, train_size=0.85,
        random_state=i, stratify=y_raw)

    for j in range(j_count):

        mlp = tf.keras.Sequential([
            tf.keras.Input(shape=(n_features,)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
        ], name='mlp')

        mlp.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        callback_0 = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=20, verbose=1,
            restore_best_weights=True)
        callback_1 = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, verbose=1,
            restore_best_weights=True)

        shot = i * j_count + j + 1
        print(f'Training of trial {shot} (i: {i}, j: {j}):')
        mlp_history = mlp.fit(
            x_train, y_train, epochs=100, batch_size=128, verbose=0,
            callbacks=[callback_0, callback_1], validation_data=(x_val, y_val),
        )
        print()

        print(f'Evaluation of trial {shot} (i: {i}, j: {j}):')
        mlp_loss, mlp_metrics = mlp.evaluate(x_test, y_test, batch_size=1)
        mlps.append((shot, mlp, mlp_loss, mlp_metrics))
        if not best_mlp_acc and not best_mlp_acc:
            best_mlp_loss, best_mlp_acc = mlp_loss, mlp_metrics
            best_shot_loss, best_shot_acc = shot, shot
        else:
            if mlp_loss < best_mlp_loss:
                best_mlp_loss, best_shot_loss = mlp_loss, shot
            if mlp_metrics > best_mlp_acc:
                best_mlp_acc, best_shot_acc = mlp_metrics, shot
        print(f'Best loss so far (trial {best_shot_loss}): '
              f'{best_mlp_loss:.4f}')
        print(f'Best accuracy so far (trial {best_shot_acc}): '
              f'{best_mlp_acc:.4f}')
        print()

        print(f'Confusion matrix of trial {shot} (i: {i}, j: {j}):')
        y_pred = np.argmax(mlp.predict(x_test), axis=1)
        print(pd.crosstab(label_classes[y_test], label_classes[y_pred],
                          rownames=['actual'], colnames=['predicted']))
        print()

    svm = False
    if svm:
        print(f'SVM (random_state: {i})')
        svm = sklearn.svm.SVC(C=200, gamma=0.0001, probability=True,
                              random_state=0)
        svm.fit(x_raw, y_raw)
        acc = svm.score(x_test, y_test)
        print(f'Score: {acc:.4f}')
        print()

    rf = False
    if rf:
        print(f'RF (random_state: {i})')
        rf = se.RandomForestClassifier(n_estimators=350, random_state=0)
        rf.fit(x_raw, y_raw)
        acc = rf.score(x_test, y_test)
        print(f'Score: {acc:.4f}')
        print()

    gbdt = False
    if gbdt:
        print(f'GBDT (random_state: {i})')
        gbdt = se.GradientBoostingClassifier(n_estimators=350, random_state=0)
        gbdt.fit(x_raw, y_raw)
        acc = gbdt.score(x_test, y_test)
        print(f'Score: {acc:.4f}')
        print()

mlps_sort_by_loss = sorted(mlps, key=lambda k: (k[2], -k[3]))
mlps_sort_by_metrics = sorted(mlps, key=lambda k: (k[3], -k[2]), reverse=True)


do_prediction = True
if do_prediction:
    print('=' * 40)
    print()
    best_mlp_candidates = mlps_sort_by_loss[:10] + mlps_sort_by_metrics[:10]
    for mlp_candidate in best_mlp_candidates:
        print(f'MLP candidate: ({mlp_candidate[0]}, {mlp_candidate[1]}, '
              f'{mlp_candidate[2]:.4f}, {mlp_candidate[3]:.4f})')
        x_raw, x_test, y_raw, y_test, f_raw, f_test = sms.train_test_split(
            features, labels, files, test_size=0.1,
            random_state=mlp_candidate[0], stratify=labels)
        x_train, x_val, y_train, y_val, f_train, f_val = sms.train_test_split(
            x_raw, y_raw, f_raw, train_size=0.85,
            random_state=mlp_candidate[0], stratify=y_raw)
        print(f'Confusion matrix of trial {mlp_candidate[0]}:')
        y_pred = np.argmax(mlp_candidate[1].predict(x_test), axis=1)
        df_pdt = pd.crosstab(label_classes[y_test], label_classes[y_pred],
                             rownames=['actual'], colnames=['predicted'])
        print(df_pdt)
        print(f'Accuracy = {np.trace(df_pdt) / len(f_test):.4f}')
        print()
        do_test_one_by_one = False
        if do_test_one_by_one:
            print('Prediction on test set one by one:')
            for f in f_test:
                features = np.array(beta_audio.feature_extraction(
                    f, dims=feature_dims, gate=feature_gate), ndmin=2)
                pdt = mlp_candidate[1](features)
                grades = [float(lc) * 11 for lc in label_classes]
                score = int(tf.tensordot(pdt, grades, axes=1).numpy()[0])
                pdt = [f'{p:.2f}' for p in pdt.numpy()[0]]
                pdt_label = [label_classes[np.argmax(pdt)]]
                print(f'{f.split(os.sep)[-2:]}: {score}, {pdt_label}, {pdt}')
            print()

print()
