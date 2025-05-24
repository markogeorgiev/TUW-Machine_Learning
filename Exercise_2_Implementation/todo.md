1. Decide on dataset:
    - Student dropout (exp: ~70% accuracy).
    - Congressional Voting (exp: I don't remember.)
2. Decide on  architecture  (2 - 3) : hidden layer, which activations. 
    - Small with 1 hidden layer with 6 neurons. 
      - Test with two different activation functions - Define which functions, so it's consistent. (ReLU, Sigmoid).
    - Slightly bigger with 4 layers (32 -> 16 -> 8 -> 4)
      - Test with two different activation functions - Define which functions, so it's consistent. (ReLU, Sigmoid)
    - At the end use add a softmax to the output layer.
    - Use `0.05` as the learning rate and n_epochs should be `100`. 
3. For _Loss_ we always use Categorical Cross Entropy.
---
- Finish the comprehensive documentation for the 'From Scratch' implementation. 
- For me TODO implement Sigmoid.