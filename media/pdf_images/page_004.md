results by 0.142 ± 0.338. Examination of rater comments shows that our neural system tends to generate speech that feels more natural and human-like, but it sometimes runs into pronunciation difficulties, e.g., when handling names. This result points to a challenge for end-to-end approaches - they require training on data that cover intended usage.

## 3.3. Ablation Studies

## 3.3.1. Predicted Features versus Ground Truth

While the two components of our model were trained separately, the WaveNet component depends on the predicted features for training. An alternative is to train WaveNet independently on mel spectrograms extracted from ground truth audio. We explore this in Table 2.

| Training     | Synthesis     | Synthesis     | Synthesis     |
|--------------|---------------|---------------|---------------|
|              | Predicted     | Ground truth  | Ground truth  |
| Predicted    | 4.526 ± 0.066 | 4.449 ± 0.060 | 4.449 ± 0.060 |
| Ground truth | 4.362 ± 0.066 | 4.522 ± 0.055 | 4.522 ± 0.055 |

Table 2. Comparison of evaluated MOS for our system when WaveNet trained on predicted/ground truth mel spectrograms are made to synthesize from predicted/ground truth mel spectrograms.

As expected, the best performance is obtained when the features used for training match those used for inference. However, when trained on ground truth features and made to synthesize from predicted features, the result is worse than the opposite. This is due to the tendency of the predicted spectrograms to be oversmoothed and less detailed than the ground truth - a consequence of the squared error loss optimized by the feature prediction network. When trained on ground truth spectrograms, the network does not learn to generate high quality speech waveforms from oversmoothed features.

## 3.3.2. Linear Spectrograms

Instead of predicting mel spectrograms, we experiment with training to predict linear-frequency spectrograms instead, making it possible to invert the spectrogram using Griffin-Lim.

| System                        | MOS           |
|-------------------------------|---------------|
| Tacotron 2 (Linear + G-L)     | 3.944 ± 0.091 |
| Tacotron 2 (Linear + WaveNet) | 4.510 ± 0.054 |
| Tacotron 2 (Mel + WaveNet)    | 4.526 ± 0.066 |

|   Total  layers |   Num  cycles |   Dilation  cycle size (samples / ms) | Receptive field  (samples / ms)   | MOS           |
|-----------------|---------------|---------------------------------------|-----------------------------------|---------------|
|              30 |             3 |                                    10 | 6,139 / 255.8                     | 4.526 ± 0.066 |
|              24 |             4 |                                     6 | 505 / 21.0                        | 4.547 ± 0.056 |
|              12 |             2 |                                     6 | 253 / 10.5                        | 4.481 ± 0.059 |
|              30 |            30 |                                     1 | 61 / 2.5                          | 3.930 ± 0.076 |

## 4. CONCLUSION

This paper describes Tacotron 2, a fully neural TTS system that combines a sequence-to-sequence recurrent network with attention to predicts mel spectrograms with a modified WaveNet vocoder. The resulting system synthesizes speech with Tacotron-level prosody and WaveNet-level audio quality. This system can be trained directly from data without relying on complex feature engineering, and achieves state-of-the-art sound quality close to that of natural human speech.

## 5. ACKNOWLEDGMENTS

The authors thank Jan Chorowski, Samy Bengio, Aïron van den Oord, and the WaveNet and Machine Hearing teams for their helpful discussions and advice, as well as Heiga Zen and the Google TTS team for their feedback and assistance with running evaluations. The authors are also grateful to the very thorough reviewers.