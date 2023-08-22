# Time-to-Pattern: Information-Theoretic Unsupervised Learning for Scalable Time Series Summarization

Time-to-Pattern (T2P) is an innovative neural network architecture that aims to identify a diverse set of patterns encapsulating the most pertinent information of the original data, in alignment with the principle of minimum description length. T2P is built as a deep generative model that learns informative embeddings of discrete time series within a specifically designed interpretable latent space.

## Technical Details

The T2P architecture consists of two main components: an encoder and a decoder.

### Encoder
The encoder commences with a series of convolutional layers. Each layer utilizes small 1D filters to scrutinize individual subsequences and extract essential features. The produced feature maps are then passed through a final convolutional layer equipped with 3D filters, specifically engineered to match the number of patterns.

### Decoder
The decoder receives the output from the encoder, referred to as `z`. It is composed of a singular deconvolution layer furnished with `n_{patterns}` filters, where the patterns of interest are found.

## Installation

Follow these steps to install and setup the project:

1. Create a new conda environment using the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

2. Once the environment is successfully created, activate it:

    ```bash
    conda activate t2p
    ```

