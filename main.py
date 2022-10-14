import numpy as np
import re
import string
import unicodedata
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers

from TextProcessing import Encode, Decode, NaturalLanguageProcessing, Attention

EPOCHS = 10
BATCH_SIZE = 64
step_per_epoch = len(x_train) // BATCH_SIZE
HM_UNITS_ENCODE = 1024
HM_UNITS_DECODE = 1024
embending_dim = 256
vocab_inp_size = len(input_key.word_index) + 1
vocab_out_size = len(output_key.word_index) + 1

persian_path = 'mizan/mizan_fa.txt'
english_path = 'mizan/mizan_en.txt'

def prepare_data(sentense):
    sentense = re.sub(r'[\n]', r'', sentense)
    sentense = re.sub(r'[?$*@#]', r'', sentense)
    sentense = '<start> ' + sentense + ' <end>'
    return sentense

def ReadText(path, maxlen=1000):
    lst = []
    with open(path, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line = prepare_data(line)
            lst.append(line)
            if i==maxlen:
                break   
    return lst


def LossFunc(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def train_step(inp, out, enc_hidden):
    loss = 0
    with tf.GradientTape() as grad:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([output_key.word_index['<start>']] * BATCH_SIZE, 1)
        for t in range(1, out.shape[1]):
            predictions, dec_hidden, _ = decode(dec_input, dec_hidden, enc_output)
            loss += LossFunc(out[:, t], predictions)
            dec_input = tf.expand_dims(out[:, t], 1)
        batch_loss = (loss / int(out.shape[1]))
        variables = encoder.trainable_variables + decode.trainable_variables
        gradients = grad.gradient(loss, variables)
        opt.apply_gradients(zip(gradients, variables))
        return batch_loss


def evaluate(sentence):
    def MaxLength(tensore):
        return max(len(t) for t in tensore)
    
    max_len_inp = MaxLength(input_value)
    max_len_out = MaxLength(output_value)
    inputs = [input_key.word_index[i] for i in sentence.split(' ')]
    inputs = prepare_data(inputs)
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_len_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    results = ''
    hidden = [tf.zeros((1, HM_UNITS_ENCODE))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([output_key.word_index['<start>']], 0)
    for t in range(max_len_out):
        predictions, dec_hidden, waigth = decode(dec_input, dec_hidden, enc_out)
        waigth = tf.reshape(waigth, (-1, ))
        prediced_id = tf.argmax(predictions[0]).numpy()
        results += output_key.index_word[prediced_id] + ' '
        if output_key.index_word[prediced_id]=='<end>':
            return results, sentence
        dec_input = tf.expand_dims([prediced_id], 0)

        return results, sentence
    

persian = ReadText(persian_path)
english = ReadText(english_path)    

nlp = NaturalLanguageProcessing(persian, english)
input_value, output_value, input_key, output_key = nlp.TokenizerText()
x_train, x_valid, y_train, y_valid = train_test_split(input_value, output_value, test_size=0.12)
print(f"Shape of X_train {x_train.shape}")
print(f"Shape of the Y_train {y_train.shape}")
print(f"Shape of the X_valid {x_valid.shape}")
print(f"Shape of the Y_valid {y_valid.shape}")

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encode(vocab_inp_size, embending_dim, HM_UNITS_ENCODE, BATCH_SIZE)
decode = Decode(vocab_out_size, embending_dim, HM_UNITS_DECODE, BATCH_SIZE)

opt = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
checkpoint = tf.train.Checkpoint(optimizer=opt, encoder=encoder, decoder=decode)

for epoch in range(EPOCHS):
    enc_hidden = encoder.InitialHiddenState()
    total_loss = 0
    for inp, out in dataset.take(step_per_epoch):
        batch_loss = train_step(inp, out, enc_hidden)
        total_loss += batch_loss
    print(f'Epochs {epoch} Total Loss is {total_loss.numpy()}')
    checkpoint.save(file_prefix='test_one')

    
checkpoint.restore(tf.train.latest_checkpoint(''))

text_test = "زن زندگی آزادی ."
translate = evaluate(text_test) # translate text
print(f"translate this sentence [[{translate}]]")
