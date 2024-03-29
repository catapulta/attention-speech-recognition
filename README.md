# Automatic Speech Recognition (speech-to-text)

Implementation based on [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)

```
The Listener (encoder) is a pyramidal recurrent network encoder that accepts filter bank spectra as inputs. The Speller (decoder) is an attention-based recurrent network decoder that emits characters as outputs. The network produces character sequences without making any independence assumptions between the characters.
```

![las](las.png)

Training objective: Predict the next phoneme in the sequence given the corresponding utterances (voice recordings) and transcripts.

Trained on the [WSJ0 dataset](https://catalog.ldc.upenn.edu/LDC93s6a)
