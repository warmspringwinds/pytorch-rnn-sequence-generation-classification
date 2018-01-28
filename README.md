# Lyrics and piano music generation in Pytorch


Implementation of generative character-level and multi-pitch-level rnn models described in "Learning to generate lyrics and music with Recurrent Neural Networks" [blog post](http://warmspringwinds.github.io/pytorch/rnns/2018/01/27/learning-to-generate-lyrics-and-music-with-recurrent-neural-networks/).

The original jupyter notebook source of the blog can be found [here](blog_post.ipynb).

Trained models can be downloaded via the following [link](https://www.dropbox.com/s/23d9n091jje8sct/music_lyrics_blogpost_models.zip?dl=0). You can skip training and
sample using provided models (follow jupyter notebooks below). Some examples of the piano song samples [are available on youtube](https://www.youtube.com/watch?v=EOQQOQYvGnw&list=PLJkMX36nfYD000TG-T59hmEgJ3ojkOlBp), and examples of lyris samples can be found in the [original blog post](http://warmspringwinds.github.io/pytorch/rnns/2018/01/27/learning-to-generate-lyrics-and-music-with-recurrent-neural-networks/).

## Lyrics Generation

We are providing jupyter notebooks for training and sampling from generative RNN-based models
trained on a [song lyrics dataset](https://www.kaggle.com/mousehead/songlyrics) which features most
popular/recent artists. Separate notebooks are avalable for:

1. Training of the unconditional RNN-based generative model on the specified lyrics dataset ([notebook file](notebooks/unconditional_lyrics_training.ipynb)).
2. Sampling from a trained unconditional RNN-based generative model ([notebook file](notebooks/unconditional_lyrics_sampling.ipynb)).
3. Training of the conditional RNN-based generative model ([notebook file](notebooks/conditional_lyrics_training.ipynb)).
4. Sampling from a trained conditional RNN-based generative model ([notebook file](notebooks/conditional_lyrics_sampling.ipynb)).

## Piano polyphonic midi song generation

We are providing jupyter notebooks for training and sampling from generative RNN-based models
trained on a [piano midi songs dataset](http://www-etud.iro.umontreal.ca/~boulanni/icml2012). Separate notebooks are avalable for:

1. Training of the RNN-based generative model on the specified piano midi dataset ([notebook file](notebooks/music_generation_training_nottingham.ipynb)).
2. Sampling from a trained RNN-based generative model ([notebook file](notebooks/music_sampling.ipynb)).