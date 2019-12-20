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