While linear spectrograms discard phase information (and are therefore lossy), algorithms such as Griffin-Lim [14] are capable of estimating this discarded information, which enables time-domain conversion via the inverse short-time Fourier transform. Mel spectrograms discard even more information, presenting a challenging inverse problem. However, in comparison to the linguistic and acoustic features used in WaveNet, the mel spectrogram is a simpler, lowerlevel acoustic representation of audio signals. It should therefore be straightforward for a similar WaveNet model conditioned on mel spectrograms to generate audio, essentially as a neural vocoder. Indeed, we will show that it is possible to generate high quality audio from mel spectrograms using a modified WaveNet architecture.

## 2.2. Spectrogram Prediction Network

As in Tacotron, mel spectrograms are computed through a shorttime Fourier transform (STFT) using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function. We experimented with a 5 ms frame hop to match the frequency of the conditioning inputs in the original WaveNet, but the corresponding increase in temporal resolution resulted in significantly more pronunciation issues.

We transform the STFT magnitude map to the mel scale using an 80 channel mel filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression. Prior to log compression, the filterbank output magnitudes are clipped to a minimum value of 0.01 in order to limit dynamic range in the logarithmic domain.

The network is composed of an encoder and a decoder with attention. The encoder converts a character sequence into a hidden feature representation which the decoder consumes to predict a spectrogram. Input characters are represented using a learned 512-dimensional character embedding, which are passed through a stack of 3 convolutional layers each containing 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters, followed by batch normalization [18] and ReLU activations. As in Tacotron, these convolutional layers model longer-term context (e.g., N -grams) in the input character sequence. The output of the final convolutional layer is passed into a single bi-directional [19] LSTM [20] layer containing 512 units (256 in each direction) to generate the encoded features.

The encoder output is consumed by an attention network which summarizes the full encoded sequence as a fixed-length context vector for each decoder output step. We use the location-sensitive attention from [21], which extends the additive attention mechanism [22] to use cumulative attention weights from previous decoder time steps as an additional feature. This encourages the model to move forward consistently through the input, mitigating potential failure modes where some subsequences are repeated or ignored by the decoder. Attention probabilities are computed after projecting input and lo- cation features to 128-dimensional hidden representations. Location features are computed using 32 I-D convolution filters of length 31.

The decoder is an autoregressive recurrent neural network which predicts a mel spectrogram from the encoded input sequence one frame at a time. The prediction from the previous time step is first passed through a small pre-net containing 2 fully connected layers of 256 hidden ReLU units. We found that the pre-net acting as an information bottleneck was essential for learning attention. The pre- net output and attention context vector are concatenated and passed through a stack of 2 uni-directional LSTM layers with 1024 units. The concatenation of the LSTM output and the attention context vector is projected through a linear transform to predict the target spectrogram frame. Finally, the predicted mel spectrogram is passed through a 5-layer convolutional post-net which predicts a residual to add to the prediction to improve the overall reconstruction. Each

post-net layer is comprised of 512 filters with shape 5 × 1 with batch normalization, followed by tanh activations on all but the final layer.

We minimize the summed mean squared error (MSE) from before and after the post-net to aid convergence. We also experimented with a log-likelihood loss by modeling the output distribution with a Mixture Density Network [23, 24] to avoid assuming a constant variance over time, but found that these were more difficult to train and they did not lead to better sounding samples.

In parallel to spectrogram frame prediction, the concatenation of decoder LSTM output and the attention context is projected down to a scalar and passed through a sigmoid activation to predict the probability that the output sequence has completed. This "stop token" prediction is used during inference to allow the model to dynamically determine when to terminate generation instead of always generating for a fixed duration. Specifically, generation completes at the first frame for which this probability exceeds a threshold of 0.5.

The convolutional layers in the network are regularized using dropout [25] with probability 0.5, and LSTM layers are regularized using zoneou [26] with probability 0.1. In order to introduce output variation at inference time, dropout with probability 0.5 is applied only to layers in the pre-net of the autoregressive decoder.

In contrast to the original Tacotron, our model uses simpler building blocks, using vanilla LSTM and convolutional layers in the encoder and decoder instead of "CBGH" stacks and GRU recurrent layers. We do not use a "reduction factor", i.e., each decoder step corresponds to a single spectrogram frame.

## 2.3. WaveNet Vocoder

We use a modified version of the WaveNet architecture from [8] to invert the mel spectrogram feature representation into time-domain waveform samples. As in the original architecture, there are 30 dilated convolution layers, grouped in 3 dilation cycles, i.e., the dilation rate of layer k ( k = 0 … 29) is 2 k (mod 10) . To work with the 12.5 ms frame hop of the spectrogram frames, only 2 upsampling layers are used in the conditioning stack instead of 3 layers.

Instead of predicting discretized buckets with a softmax layer, we follow PixelCNN++ [27] and Parallel WaveNet [28] and use a 10component mixture of logistic distributions (Mol) to generate 16-bit samples at 24 kHz. To compute the logistic mixture distribution, the WaveNet stack output is passed through a ReLU activation followed

![page_002_picture_01.png](my_streamlit_app/media/crops/page_002_picture_01.png)

**Image page_002_picture_01.png:** Okay, let's break down the image you sent, which illustrates a diagram of a WaveNet MoLe (Multi-Level) model architecture for speech recognition. Here's a detailed description:

**Overall Structure:**

The image depicts a neural network architecture designed for speech recognition, combining both audio (waveform) and textual information. It's a hybrid model, leveraging the strengths of both modalities.

**Components:**

1. **Input Text:**
   -  At the bottom left, there's a box labeled "Input Text." This indicates that the model initially receives textual input, likely a transcription of the speech.

2. **Character Embedding:**
   - This input text is then fed into a "Character Embedding" layer.  This layer converts the individual characters of the input text into numerical representations, which the network can then process.

3. **5 Conv Layer Post-Net:**
   - This is one of the key components. A "5 Conv Layer Post-Net" takes in the output from the other layers and it's a Convolutional Neural Network (CNN) that has 5 convolutional layers and is used for sequence modeling. 

4. **2 Layer Pre-Net:**
   - Next is the "2 Layer Pre-Net", which is a CNN used for time-series modeling and sequence modeling. 

5. **Bidirectional LSTM:**
   - A "Bidirectional LSTM" layer is used to model the temporal dynamics of the speech data. Bidirectional LSTMs process the input sequence both forwards and backwards in time, allowing the model to understand context from both past and future frames.

6. **Linear Projection:**
   - Two "Linear Projection" layers are present. These layers are used to map the output of the LSTM layers to a lower dimensional space, often for compatibility with other layers.

7. **Location Sensitive Attention:**
   - A "Location Sensitive Attention" layer. This is a crucial element that incorporates information about the location (frame number) of the audio data. This helps the model understand how the audio signal changes over time and learn more effectively.

8.  **Mel Spectrogram:**
    - The main input of the model is the "Mel Spectrogram," which is a visual representation of the audio spectrum, reflecting the energy at different frequencies over time. This spectrogram is the raw audio data that the model learns from.

9. **WaveNet MoLe:**
    -  This indicates the model’s name, “WaveNet MoLe.” This shows that it is built on the WaveNet architecture and leverages Multi-Level processing.

10. **Stop Token:**
    -  A "Stop Token" is used to signal the end of the sequence, prompting the model to produce the final output.

**Connections and Flow:**

*   The components are connected in a sequential manner, indicating the flow of information through the model.
*   The output from each layer is fed into the next, allowing the model to build a representation of the speech sequence.

**Key Concepts:**

*   **WaveNet:** This architecture is based on WaveNet, a generative model for speech synthesis that uses dilated convolutions to capture long-range dependencies in audio signals.
*   **Multi-Level:**  The "MoLe" part of "WaveNet MoLe" suggests that the model processes the audio at different time scales, potentially combining information from both short and long-range dependencies. 
*   **Hybrid Approach:** The combination of text and audio data enhances the model's ability to recognize speech, especially in noisy environments or when the textual context is available.

Do you want me to delve deeper into a specific aspect of the diagram, such as the function of the dilated convolutions, or how the hybrid approach works?

