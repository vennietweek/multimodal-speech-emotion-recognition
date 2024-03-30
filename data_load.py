from collections import Counter
import pickle
import numpy as np

# emotion_counter = Counter()

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

# Load data
def load_data():
    f = open("dataset/iemocap/raw/IEMOCAP_features_raw.pkl", "rb")
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
    # label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}

    print('Number of training samples: ',len(trainVid))
    print('Number of testing samples: ',len(testVid))

    train_audio = []
    train_text = []
    train_visual = []
    train_seq_len = []
    train_label = []

    test_audio = []
    test_text = []
    test_visual = []
    test_seq_len = []
    test_label = []
    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))

    max_len = max(max(train_seq_len), max(test_seq_len))
    print('Max length of sequences: ', max_len)
    print()

    for vid in trainVid:
        # emotion_counter.update(videoLabels[vid])
        train_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
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
        test_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

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

    return train_data, train_label, test_data, test_label, train_text, train_audio, train_visual, test_text, test_audio, test_visual