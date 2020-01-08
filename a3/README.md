# Assignment 3: Dependency Parser

## 1. Machine Learning & Neural Networks

(a) Adam Optimizer

i)

$m$ keeps previous direction of the gradient, and by using the momentum we could hold a more steady course towards the minima. 

ii)

TO-DO

(b) 

i)

$$
E_{p_{drop}}[h_{drop}] = h \\
E_{p_{drop}}[\gamma d \odot h] = h \\
\gamma*(1-p)*h = h \\
\gamma = \frac{1}{1-p}$$  

ii)

The purpose of dropout is that the redundent hidden values behave like model ensembles in evaluation phase which can help reduce the model variance. 

## 2. Neural Transition-Based Dependency Parsing
(a)

|Stack|Buffer|New dependency|Transition|
|:-|:-|:-|:-|
|[ROOT]|[I, parsed, this, sentence, correctly]||Initial Configuration
|[ROOT, I]|[parsed, this, sentence, correctly]||SHIFT
|[ROOT, I, parsed]|[this, sentence, correctly]||SHIFT
|[ROOT, parsed]|[this, sentence, correctly]|parsed->I|LEFT_ARC
|[ROOT, parsed, this]|[sentence, correctly]||SHIFT
|[ROOT, parsed, this, sentence]|[correctly]||SHIFT
|[ROOT, parsed, sentence]|[correctly]|sentence->this|LEFT_ARC
|[ROOT, parsed]|[correctly]|parsed->sentence|RIGHT_ARC
|[ROOT, parsed, correctly]|[]||SHIFT
|[ROOT, parsed]|[]|parsed->correctly|RIGHT_ARC
|[parsed]|[]|ROOT->parsed|RIGHT_ARC

(b)

The number of steps is $2\times n$, because each time either the buffer is decreased by 1 or the stack is decreased by 1.