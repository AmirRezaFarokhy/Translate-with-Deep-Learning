import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# input data is persian and output or target data is english
class NaturalLanguageProcessing:
    
    def __init__(self, persian, english):
        self.input = persian
        self.output = english

    def Tokenizer(self, language):
        key_tokens = tf.keras.preprocessing.text.Tokenizer(filters='')
        key_tokens.fit_on_texts(language)
        value_tensor = key_tokens.texts_to_sequences(language)
        value_tensor = tf.keras.preprocessing.sequence.pad_sequences(value_tensor, padding='post')
        return key_tokens, value_tensor

    def TokenizerText(self):
        self.input_key, self.input_value = self.Tokenizer(self.input)
        self.output_key, self.output_value = self.Tokenizer(self.output)
        return self.input_value, self.output_value, self.input_key, self.output_key
    
    def ShowSamples(self, key, value):
        for i in key:
            if i!=0:
                print(i, value.index_word[i])
                
                
class Encode(tf.keras.Model):
    
    def __init__(self, vocab_size, embending_dim, hm_unit_en, batch_size):
        super(Encode, self).__init__()
        self.batch_size = batch_size
        self.hm_unit_en = hm_unit_en
        self.embending = tf.keras.layers.Embedding(vocab_size, embending_dim)
        self.gru = tf.keras.layers.GRU(self.hm_unit_en, return_sequences=True, return_state=True)
        
    def call(self, x, hidden):
        x = self.embending(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state 
    
    def InitialHiddenState(self):
        return tf.zeros((self.batch_size, self.hm_unit_en))
    

class Attention(tf.keras.layers.Layer):
    
    def __init__(self, unints):
        super(Attention, self).__init__()
        self.w1 = tf.keras.layers.Dense(unints) 
        self.w2 = tf.keras.layers.Dense(unints)
        self.v = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        hidden_axis = tf.expand_dims(query, 1)
        score = self.v(tf.nn.tanh(self.w1(values) + self.w2(hidden_axis)))
        waights = tf.nn.softmax(score, axis=1)
        context_vectore = waights * values
        context_vectore = tf.reduce_sum(context_vectore, axis=1)
        return context_vectore, waights
    

class Decode(tf.keras.Model):
    
    def __init__(self, vocab_size, embending_dim, hm_unit_de, batch_size):
        super(Decode, self).__init__()
        self.batch_size = batch_size
        self.hm_unit_de = hm_unit_de
        self.embending = tf.keras.layers.Embedding(vocab_size, embending_dim)
        self.gru = tf.keras.layers.GRU(self.hm_unit_de, return_sequences=True, return_state=True)
        self.forcast = tf.keras.layers.Dense(vocab_size)
        self.attentions = Attention(self.hm_unit_de)
    
    def call(self, x, hidden, enc_output):
        context_vector, waigth = self.attentions(hidden, enc_output)
        x = self.embending(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.forcast(output)
        return x, state, waigth
    
    
    
