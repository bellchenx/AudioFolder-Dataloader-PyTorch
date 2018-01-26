# Introduction
In this repository, I inplementd AudioFolder function with data-loader and a full convolutional classifier for classical music. The AudioFolder function was included in "dataloader.py". And "batchspliter.py" can transfer normal music files to short audio splits which matches the format as training dataset.

Regretfully, after splitting audio, you need to manually select training data and testing data and put them in order like "train/a", "train/b", "test/a", and "test/b".

The including classifier is just a sample for data-loader function, so it works really bad for music classification. If you want to build up a more effective classifier, please use LSTM-RNN, sampleRNN and so on.
