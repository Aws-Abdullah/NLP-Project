import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#.keras file contains weights, layers, shapes ....., attention class needs to be redefined to load model or it will error also prepeocess the text because it was trained like this 

class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, inputs):
        encoder_out, decoder_out = inputs
        enc_exp = tf.expand_dims(encoder_out, 1)
        dec_exp = tf.expand_dims(decoder_out, 2)
        score = self.V(tf.nn.tanh(self.W1(enc_exp) + self.W2(dec_exp)))
        score = tf.squeeze(score, -1)
        attn_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(attn_weights, encoder_out)
        return context, attn_weights


_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
_tatweel = re.compile(r'\u0640')
_non_arabic = re.compile(r'[^\u0600-\u06FF\s]')
_extra_spaces = re.compile(r'\s+')


def clean_arabic(text):
    text = _diacritics.sub('', text)
    text = _tatweel.sub('', text)
    text = _non_arabic.sub(' ', text)
    text = _extra_spaces.sub(' ', text)
    return text.strip()


def normalize_arabic(text):
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ؤ', 'ء', text)
    text = re.sub('ئ', 'ء', text)
    text = re.sub('ة', 'ه', text)
    return text


def preprocess_text(text):
    return normalize_arabic(clean_arabic(text))


def build_models(vocab_size, embedding_dim, latent_dim, max_article_len):
    encoder_inputs = Input(shape=(max_article_len,), name='encoder_input')
    enc_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)

    enc_lstm = Bidirectional(
        LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.2)
    )
    enc_out, fwd_h, fwd_c, bwd_h, bwd_c = enc_lstm(enc_emb)

    state_h = Concatenate()([fwd_h, bwd_h])
    state_c = Concatenate()([fwd_c, bwd_c])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dec_emb_layer = Embedding(vocab_size, embedding_dim, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, dropout=0.2)
    dec_out, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    attention = BahdanauAttention(latent_dim)
    context, _ = attention([enc_out, dec_out])

    decoder_concat = Concatenate(axis=-1)([dec_out, context])
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat)

    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Inference encoder
    encoder_model = Model(encoder_inputs, [enc_out, state_h, state_c])

    # Inference decoder
    dec_state_h_in = Input(shape=(latent_dim * 2,))
    dec_state_c_in = Input(shape=(latent_dim * 2,))
    dec_enc_out_in = Input(shape=(max_article_len, latent_dim * 2))

    dec_emb_inf = dec_emb_layer(decoder_inputs)
    dec_out_inf, h_inf, c_inf = decoder_lstm(
        dec_emb_inf, initial_state=[dec_state_h_in, dec_state_c_in]
    )
    context_inf, _ = attention([dec_enc_out_in, dec_out_inf])
    dec_concat_inf = Concatenate(axis=-1)([dec_out_inf, context_inf])
    dec_outputs_inf = decoder_dense(dec_concat_inf)

    decoder_model = Model(
        [decoder_inputs, dec_enc_out_in, dec_state_h_in, dec_state_c_in],
        [dec_outputs_inf, h_inf, c_inf]
    )

    return train_model, encoder_model, decoder_model


def load_resources(model_path, tokenizer_path, config_path):
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    train_model, encoder_model, decoder_model = build_models(
        config['vocab_size'],
        config['embedding_dim'],
        config['latent_dim'],
        config['max_article_len']
    )

    train_model.load_weights(model_path)

    return train_model, encoder_model, decoder_model, tokenizer, config


def generate_summary(text, encoder_model, decoder_model, tokenizer, config):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=config['max_article_len'], padding='post', truncating='post')

    enc_out_val, h, c = encoder_model.predict(padded, verbose=0)
    target = np.array([[config['sos_id']]])

    oov_id = tokenizer.word_index.get('<OOV>', -1)
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    index_to_word[0] = ''

    decoded = []
    for _ in range(config['max_summary_len']):
        probs, h, c = decoder_model.predict([target, enc_out_val, h, c], verbose=0)
        p = probs[0, -1, :].copy()
        if oov_id != -1:
            p[oov_id] = 0
        p[0] = 0

        token_id = int(np.argmax(p))
        if token_id == config['eos_id']:
            break

        word = index_to_word.get(token_id, '')
        if word and word not in ('sos', 'eos'):
            decoded.append(word)

        target = np.array([[token_id]])

    return ' '.join(decoded)