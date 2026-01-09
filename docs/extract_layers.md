# Extract layers

Back to home: [README](../README.md)

Navigate to src directory then:
```bash
python -m extract_layers.extract_layers_main
```

This will:
- Load your model from HuggingFace
- Process all training data
- Extract pooled representations (mean + std) from each layer
- Save to `representations/layer_X.npy`

Currently, the pooling is done along the sequence dimension: given the hidden layer activations of shape 
(N=number of samples, L=sequence length, D=hidden dim), the pooled result is (N, 2D) = (N, D) for mean + (N, D) for std.

If this is not effective, then there will be other pooling methods coded. 

Source code for this step is in the folder extract_layers/