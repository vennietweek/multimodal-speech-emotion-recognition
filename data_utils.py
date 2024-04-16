from collections import Counter
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createOneHot(train_label, test_label):
    
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test


def load_data():
    f = open("dataset/iemocap/raw/IEMOCAP_features_raw.pkl", "rb")
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
    new_label_mapping = {0: 1, 4: 1, 3: 2, 5: 2, 1: 3, 2: 4}
    # mapping = {1: 'happy', 2: 'angry', 3: 'sad', 4: 'neutral'}
    
    train_audio, train_text, train_visual, train_label, train_seq_len = [], [], [], [], []
    test_audio, test_text, test_visual, test_label, test_seq_len = [], [], [], [], []

    # Get the maximum sequence length
    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))
    max_len = max(max(train_seq_len), max(test_seq_len))
    print('Max length of sequences: ', max_len)
    print()

    for vid in trainVid:
        updated_labels = [new_label_mapping[label] for label in videoLabels[vid]]
        train_label.append(updated_labels + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        train_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        train_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        train_visual.append(video)

    for vid in testVid:
        updated_labels = [new_label_mapping[label] for label in videoLabels[vid]]
        test_label.append(updated_labels + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

    print('Number of training samples:', len(trainVid))
    print('Number of testing samples:', len(testVid))
    print()

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    print(f"Train text shape: {train_text.shape[0]} samples, {train_text.shape[1]} timesteps, {train_text.shape[2]} features")
    print(f"Train audio shape: {train_audio.shape[0]} samples, {train_audio.shape[1]} timesteps, {train_audio.shape[2]} features")
    print(f"Train visual shape: {train_visual.shape[0]} samples, {train_visual.shape[1]} timesteps, {train_visual.shape[2]} features")
    print()

    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)
    print(f"Test text shape: {test_text.shape[0]} samples, {test_text.shape[1]} timesteps, {test_text.shape[2]} features")
    print(f"Test audio shape: {test_audio.shape[0]} samples, {test_audio.shape[1]} timesteps, {test_audio.shape[2]} features")
    print(f"Test visual shape: {test_visual.shape[0]} samples, {test_visual.shape[1]} timesteps, {test_visual.shape[2]} features")
    print()

    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)

    train_mask = np.zeros((train_text.shape[0], train_text.shape[1]), dtype='float')
    for i in range(len(train_seq_len)):
        train_mask[i, :train_seq_len[i]] = 1.0

    test_mask = np.zeros((test_text.shape[0], test_text.shape[1]), dtype='float')
    for i in range(len(test_seq_len)):
        test_mask[i, :test_seq_len[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)
    train_data = np.concatenate((train_audio, train_visual, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_visual, test_text), axis=-1)

    return train_data, train_label, train_mask, test_data, test_label, test_mask, train_text, train_audio, train_visual, test_text, test_audio, test_visual

def prepare_dataset(audio=None, visual=None, text=None, data=None, labels=None, mask=None, 
                    modality='unimodal', mode='train', batch_size=32, train_size=0.8, shuffle_size=120):

    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    if modality == 'unimodal':
        inputs = tf.convert_to_tensor(audio, dtype=tf.float32) if audio is not None else None
        inputs = tf.convert_to_tensor(visual, dtype=tf.float32) if visual is not None else None
        inputs = tf.convert_to_tensor(text, dtype=tf.float32) if text is not None else None
        inputs = tf.convert_to_tensor(data, dtype=tf.float32) if data is not None else None
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, mask))

    elif modality == 'multimodal':
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        visual = tf.convert_to_tensor(visual, dtype=tf.float32)
        text = tf.convert_to_tensor(text, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(((audio, visual, text), labels, mask))
    
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=shuffle_size)
        total_size = labels.shape[0]
        train_size = int(train_size * total_size)
        train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset, val_dataset
    
    elif mode == 'test':
        test_dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset
