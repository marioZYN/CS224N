# Assignment 4: NMT
## 1. Neural Machine Translation with RNNs
(g) It is useless to take into account \<pad> words when we are calculating attention distribution. By using enc_masks, we make all \<pad> words have -inf value. Thus, \<pad> words will have zero weights when we compute the attention output. If we do not use enc_masks, these \<pad> words may mislead attention output.  
(i) BLEU: 22.50  
(j) difference between dot product attention, multiplicative attention, addictive attention
* dot product attention: 
    * advantage: simple and easy to use
    * disadvantage: s_t and h_i must have the same dimension
* multiplicative attention:
    * advantage: add a transition matrix compared to dot product attention, thus more powerful
    * disadvantage: more parameters to learn, treat h_t and s_t equally
* addictive attention:
    * advantage: more powerful than the previous two
    * disadvantage: more parameters to learn, and slow in compution when matrix dimension is large  

## 2. Analyzing NMT systems
(a)  
_[i]. (2 points) Source Sentence: Aqu´ı otro de mis favoritos, “La noche estrellada”.  
Reference Translation: So another one of my favorites, “The Starry Night”.   
NMT Translation: Here’s another favorite of my favorites, “The Starry Night”._

_[ii]. (2 points) Source Sentence: Ustedes saben que lo que yo hago es escribir para los ni˜nos, y, de hecho, probablemente soy el autor para ni˜nos, ms ledo en los EEUU.  
Reference Translation: You know, what I do is write for children, and I’m probably America’s most widely read children’s author, in fact.  
NMT Translation: You know what I do is write for children, and in fact, I’m probably the author for children, more reading in the U.S._

_[iii]. (2 points) Source Sentence: Un amigo me hizo eso – Richard Bolingbroke.  
Reference Translation: A friend of mine did that – Richard Bolingbroke.  
NMT Translation: A friend of mine did that – Richard \<unk>_

> Then \<unk> error comes from the out-of-vocab. We can enlarge our dictionary to overcome this.

_[iv, v, vi] TO-DO_

(b) TO-DO

(c)  
(i)  
for c1:   
p1 = 0.6  
p2 = 0.5  
c = 5  
r* = 4  
BP = 1  
BLEU = 0.77  
for c2:  
p1 = 0.8  
p2 = 0.5  
c = 5  
r* = 4  
BP = 1  
BLEU = 0.82  
According to the BLEU score, the second one is better. I agree with it.  
(ii)  
for c1:   
p1 = 0.6  
p2 = 0  
c = 5  
r* = 6  
BP = 0.8187  
BLEU = 0.73  
for c2:  
p1 = 0.4  
p2 = 0.2  
c = 5  
r* = 6  
BP = 0.8187  
BLEU = 0.47  
The first one receives higher BLEU score. I do not agree with it.  
(iii)  
With only one reference sentence, the n-gram selections may be too specific. The BLEU metric favors the answers having more n-grams appeared in the reference, but the whole translation can be bad.  
(iv)  
advantage: 
* easy to compute
* reasonable to some extent

disadvantage:  
* not good when references are few
* only favors n-gram apperance, not the meaning
