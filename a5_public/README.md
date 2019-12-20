# Assignment 5: NMT with char decoder
## 1. Character-based convolutional encoder for NMT
(a) Because the total amount of char characters is a lot less than words.

(b) 
* characer-based embedding model = (V_char * e_char) + (k * e_word * e_char + e_word)+2 * (e_word * e_word + e_word)
* word-based lookup model = e_word * V_word  

Word-based lookup model has more parameters, about thousand times more.

(c) cnn has fewer parameters than lstm which makes it easier and faster to train. The number of parameters in cnn is num_filters * e_char * k, while the number of a uni-directional lstm is 4 * (hidden_size * hidden_size + hidden_size * embed_size)

(d) max pooling selects the most significant value while average pooling smooth all the values.
* max pooling: keeps the most significant value, but can lose some info
* average pooling: keeps all the info, but can have noise

## 3. Analyzing NMT Systems
(a) 
* occur: traducir, traduce
* not occur: traduzco, traduces, traduzca, traduzcas

In word-based NMT, we can not predict words which didn't appear in the word dictionary, thus the translation quality is limited by the quality of the word dic.

In character-aware NMT model, we can predict \<unk\> word.

(b) orignal - word2vec - charEmbed
* finacial - economic - 
* neuron - nerve - Newton
* Francisco - san - France
* naturally - occuring - pracitcally
* expectation - norms - exception

(c)
* word2vec: semantic similarity
* charcnn: structure similarity

Word2vec is trained under the hyposis that similar words have similar context, thus it learns the semantic meaning of each word. In most times, given a sentence we could substitude the original word with the nearest neighbour word.

CharCnn learns the word embeddings from convolution window, thus it preserves word structures. 