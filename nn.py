"""
Neural Network Implementation from Scratch

This module implements a complete neural network from scratch using only pure Python
and basic mathematical operations. It's designed for educational purposes to help
beginners understand the fundamental concepts of neural networks and the mathematics
behind them.

Mathematical Foundation:
- Matrix operations (dot product, transpose, element-wise operations)
- Activation functions (ReLU, Softmax)
- Loss functions (Categorical Cross-Entropy)
- Forward propagation

Author: Educational Implementation
Purpose: Learning neural networks from first principles
"""

import math
import random
from typing import List

from create_data import create_data


class NeuralMatrix:
    """
    Matrix operations library for neural network computations.
    
    This class implements all necessary matrix operations using pure Python
    with nested lists representing matrices. Each method is designed to be
    educational and easy to understand.
    
    Mathematical Concepts:
    - Matrices are represented as nested lists: [[row1], [row2], ...]
    - Vectors are represented as single lists: [element1, element2, ...]
    - All operations follow standard linear algebra rules
    """

    def is_scalar_list(self, data: List) -> bool:
        """
        Check if the input is a list of scalars (1D) or list of lists (2D).
        
        Args:
            data: Input list to check
            
        Returns:
            True if data contains scalars, False if it contains sublists
            
        Mathematical Context:
        This helps distinguish between vectors [a, b, c] and matrices [[a, b], [c, d]]
        """
        return not isinstance(data[0], list)

    def sum_rows(self, matrix: List[List[float]]) -> List[float]:
        """
        Calculate the sum of each row in a matrix.
        
        Mathematical Formula: For matrix M[i,j], compute Σ(M[i,j]) for each row i
        
        Args:
            matrix: Input matrix as list of lists
            
        Returns:
            List containing sum of each row
            
        Example:
            Input: [[1, 2, 3], [4, 5, 6]]
            Output: [6, 15]  # [1+2+3, 4+5+6]
        """
        row_sums = []
        for row in matrix:
            row_sum = sum(row)  # More efficient than manual loop
            row_sums.append(row_sum)
        return row_sums

    def argmax_rows(self, matrix: List[List[float]]) -> List[int]:
        """
        Find the index of maximum value in each row.
        
        Mathematical Context:
        argmax(x) = index i where x[i] is maximum
        This is crucial for classification - finding the predicted class
        
        Args:
            matrix: Input matrix
            
        Returns:
            List of indices of maximum values for each row
            
        Example:
            Input: [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]
            Output: [1, 0]  # Second element in first row, first element in second row
        """
        indices = []
        for row in matrix:
            max_index = row.index(max(row))
            indices.append(max_index)
        return indices

    def element_wise_multiply(self, matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
        """
        Element-wise multiplication of two matrices (Hadamard product).
        
        Mathematical Formula: C[i,j] = A[i,j] × B[i,j]
        
        Args:
            matrix1: First matrix
            matrix2: Second matrix
            
        Returns:
            Result of element-wise multiplication
            
        Note: Matrices must have the same dimensions
        """
        result = []
        for i in range(len(matrix1)):
            row_result = []
            for j in range(len(matrix1[i])):
                row_result.append(matrix1[i][j] * matrix2[i][j])
            result.append(row_result)
        return result

    def calculate_mean(self, matrix: List[List[float]]) -> float:
        """
        Calculate the overall mean of all elements in a matrix.
        
        Mathematical Formula: mean = (Σ all elements) / (total count)
        
        Args:
            matrix: Input matrix
            
        Returns:
            Mean value of all elements
        """
        total_sum = 0
        total_count = 0
        for row in matrix:
            total_sum += sum(row)
            total_count += len(row)
        return total_sum / total_count if total_count > 0 else 0

    def transpose(self, matrix: List[List[float]]) -> List[List[float]]:
        """
        Transpose a matrix (swap rows and columns).
        
        Mathematical Definition: If A is m×n, then A^T is n×m where A^T[i,j] = A[j,i]
        
        This operation is fundamental in neural networks for:
        - Weight matrix operations
        - Backpropagation calculations
        - Dimension matching in matrix multiplication
        
        Args:
            matrix: Input matrix to transpose
            
        Returns:
            Transposed matrix
            
        Example:
            Input:  [[1, 2, 3],    Output: [[1, 4],
                     [4, 5, 6]]             [2, 5],
                                            [3, 6]]
        """
        if not matrix or not matrix[0]:
            return []
        
        rows = len(matrix)
        cols = len(matrix[0])
        transposed = []
        
        for col in range(cols):
            new_row = []
            for row in range(rows):
                new_row.append(matrix[row][col])
            transposed.append(new_row)
        
        return transposed

    def matrix_multiply(self, inputs: List[List[float]], weights: List[List[float]], biases: List[float]) -> List[List[float]]:
        """
        Perform matrix multiplication and add bias: output = inputs × weights + biases
        
        Mathematical Formula:
        For each sample i and neuron j:
        output[i][j] = Σ(inputs[i][k] × weights[k][j]) + biases[j]
        
        This is the core operation in neural networks:
        - inputs: batch of input samples
        - weights: learnable parameters connecting layers
        - biases: learnable offset parameters
        
        Matrix Dimensions:
        - inputs: [batch_size, input_features]
        - weights: [input_features, output_neurons]  
        - biases: [output_neurons]
        - output: [batch_size, output_neurons]
        
        Args:
            inputs: Input matrix [batch_size, input_features]
            weights: Weight matrix [input_features, output_neurons]
            biases: Bias vector [output_neurons]
            
        Returns:
            Output matrix [batch_size, output_neurons]
        """
        # Dimension validation
        if len(inputs[0]) != len(weights):
            raise ValueError(
                f"Dimension mismatch: inputs have {len(inputs[0])} features, "
                f"but weights expect {len(weights)} features"
            )

        batch_size = len(inputs)
        num_neurons = len(biases)
        output = []
        
        # For each sample in the batch
        for sample_idx in range(batch_size):
            sample_output = []
            
            # For each output neuron
            for neuron_idx in range(num_neurons):
                # Compute dot product: Σ(input[k] × weight[k][neuron])
                dot_product = 0
                for feature_idx in range(len(inputs[sample_idx])):
                    dot_product += inputs[sample_idx][feature_idx] * weights[feature_idx][neuron_idx]
                
                # Add bias and store result
                final_output = dot_product + biases[neuron_idx]
                sample_output.append(final_output)
            
            output.append(sample_output)
        
        return output

    def random_matrix(self, rows: int, cols: int, scale: float = 0.1) -> List[List[float]]:
        """
        Generate a random matrix with improved initialization.
        
        Weight Initialization Strategy:
        Uses small random values scaled by 'scale' parameter to prevent:
        - Vanishing gradients (weights too small)
        - Exploding gradients (weights too large)
        
        Mathematical Justification:
        Small random weights around 0 help maintain gradient flow during training
        and prevent symmetry breaking issues.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            scale: Scaling factor for random values
            
        Returns:
            Random matrix with values in range [-scale, scale]
        """
        matrix = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                # Generate random value in range [-scale, scale]
                random_value = (random.random() - 0.5) * 2 * scale
                row.append(random_value)
            matrix.append(row)
        return matrix

    def exp_with_stability(self, matrix: List[List[float]]) -> List[List[float]]:
        """
        Compute exponential of matrix elements with numerical stability.
        
        Mathematical Problem:
        e^x can become very large, causing numerical overflow.
        
        Solution - Numerical Stability Trick:
        e^(x - max(x)) = e^x / e^max(x)
        This prevents overflow while preserving relative magnitudes.
        
        Mathematical Proof:
        If we have values [x1, x2, ..., xn] and M = max(xi), then:
        e^(xi - M) = e^xi / e^M
        
        This is crucial for softmax computation where we need:
        softmax(xi) = e^xi / Σe^xj = e^(xi-M) / Σe^(xj-M)
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix with exp applied element-wise, numerically stable
        """
        result = []
        for row in matrix:
            # Find maximum value in this row for stability
            max_val = max(row)
            exp_row = []
            
            # Apply exp(x - max) to each element
            for val in row:
                stable_exp = math.exp(val - max_val)
                exp_row.append(stable_exp)
            
            result.append(exp_row)
        
        return result

    def normalize_rows(self, matrix: List[List[float]]) -> List[List[float]]:
        """
        Normalize each row so that elements sum to 1.
        
        Mathematical Formula: 
        normalized[i][j] = matrix[i][j] / Σ(matrix[i][k]) for all k
        
        This creates a probability distribution where:
        - All values are positive
        - Each row sums to 1.0
        - Used in softmax activation function
        
        Args:
            matrix: Input matrix
            
        Returns:
            Row-normalized matrix (each row sums to 1)
        """
        normalized = []
        for row in matrix:
            row_sum = sum(row)
            if row_sum == 0:
                # Handle edge case: if row sum is 0, distribute equally
                normalized_row = [1.0 / len(row)] * len(row)
            else:
                normalized_row = [val / row_sum for val in row]
            normalized.append(normalized_row)
        return normalized

    def clip_values(self, values: List[float], min_val: float = 1e-7, max_val: float = 1 - 1e-7) -> List[float]:
        """
        Clip values to prevent numerical instability in logarithms.
        
        Mathematical Motivation:
        log(0) = -∞ and log(values very close to 0) can cause numerical issues.
        
        In loss calculations, we often need log(predicted_probability).
        To prevent log(0), we clip probabilities to be within [ε, 1-ε]
        where ε is a small positive number (1e-7).
        
        Args:
            values: Input values to clip
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Clipped values within [min_val, max_val]
        """
        return [max(min_val, min(val, max_val)) for val in values]


class ActivationFunctions:
    """
    Implementation of activation functions used in neural networks.
    
    Activation functions introduce non-linearity into neural networks,
    allowing them to learn complex patterns and relationships in data.
    
    Without activation functions, neural networks would be limited to
    learning only linear relationships, no matter how many layers they have.
    """

    def relu(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Rectified Linear Unit (ReLU) activation function.
        
        Mathematical Formula: f(x) = max(0, x)
        
        Properties:
        - Simple and computationally efficient
        - Helps solve vanishing gradient problem
        - Introduces sparsity (many outputs are exactly 0)
        - Most commonly used activation in hidden layers
        
        Mathematical Intuition:
        - For positive inputs: output = input (linear behavior)
        - For negative inputs: output = 0 (creates sparsity)
        
        Why ReLU works well:
        1. Computational efficiency: just a comparison and selection
        2. Sparse activation: many neurons output 0, creating sparse representations
        3. Unbounded for positive values: allows gradients to flow freely
        
        Args:
            inputs: Input matrix from previous layer
            
        Returns:
            Matrix with ReLU applied element-wise
            
        Example:
            Input:  [[-2, -1, 0, 1, 2]]
            Output: [[ 0,  0, 0, 1, 2]]
        """
        activated = []
        for row in inputs:
            activated_row = []
            for value in row:
                # ReLU: max(0, x) - keep positive values, zero out negative ones
                activated_value = max(0, value)
                activated_row.append(activated_value)
            activated.append(activated_row)
        return activated

    def softmax(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Softmax activation function for multi-class classification.
        
        Mathematical Formula:
        softmax(xi) = e^xi / Σ(e^xj) for all j
        
        Properties:
        - Converts any real-valued vector into a probability distribution
        - All outputs are in range (0, 1)
        - All outputs sum to 1.0
        - Used in output layer for multi-class classification
        
        Mathematical Intuition:
        The softmax function takes a vector of real numbers and transforms it into
        a probability distribution. Larger input values get higher probabilities,
        but all values remain positive and sum to 1.
        
        Steps:
        1. Apply exponential to all elements (makes them positive)
        2. Normalize by the sum (makes them sum to 1)
        
        Why use exponential?
        - Exponential amplifies differences: larger inputs get much larger outputs
        - Always positive: creates valid probabilities
        - Smooth and differentiable: good for gradient-based optimization
        
        Numerical Stability:
        We use the exp_with_stability method to prevent overflow when dealing
        with large input values.
        
        Args:
            inputs: Raw logits from final layer
            
        Returns:
            Probability distribution (each row sums to 1)
            
        Example:
            Input:  [[1, 2, 3]]
            Step 1: [[e^1, e^2, e^3]] = [[2.72, 7.39, 20.09]]
            Step 2: Normalize by sum (30.19): [[0.09, 0.24, 0.67]]
            Output: [[0.09, 0.24, 0.67]]  # Sums to 1.0
        """
        matrix_ops = NeuralMatrix()
        
        # Step 1: Apply exponential with numerical stability
        exponentials = matrix_ops.exp_with_stability(inputs)
        
        # Step 2: Normalize each row to create probability distribution
        probabilities = matrix_ops.normalize_rows(exponentials)
        
        return probabilities


class DenseLayer:
    """
    Dense (Fully Connected) Layer implementation.
    
    A dense layer connects every input to every output through learnable weights.
    This is the most fundamental type of layer in neural networks.
    
    Mathematical Operation:
    output = input × weights + biases
    
    Architecture:
    - Input: [batch_size, n_input_features]
    - Weights: [n_input_features, n_neurons] 
    - Biases: [n_neurons]
    - Output: [batch_size, n_neurons]
    
    Learning Parameters:
    - Weights: determine the strength of connection between inputs and neurons
    - Biases: provide offset values that shift the activation function
    """
    
    def __init__(self, n_input_features: int, n_neurons: int):
        """
        Initialize a dense layer with random weights and zero biases.
        
        Weight Initialization Strategy:
        - Random small values prevent symmetry breaking
        - Small magnitude prevents exploding gradients initially
        - Normal distribution around 0 helps with optimization
        
        Mathematical Justification for Weight Initialization:
        If weights are too large: gradients can explode
        If weights are too small: gradients can vanish
        Random initialization breaks symmetry - otherwise all neurons learn the same features
        
        Args:
            n_input_features: Number of input features from previous layer
            n_neurons: Number of neurons in this layer
        """
        self.n_input_features = n_input_features
        self.n_neurons = n_neurons
        
        # Initialize matrix operations helper
        self.matrix_ops = NeuralMatrix()
        
        # Initialize weights with small random values
        # Using improved initialization with scaling
        self.weights = self.matrix_ops.random_matrix(n_input_features, n_neurons, scale=0.1)
        
        # Initialize biases to zero (common practice)
        # Biases can start at zero because weights break symmetry
        self.biases = [0.0] * n_neurons
        
        # Store layer output for potential use in backpropagation
        self.output = None
        
    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Forward propagation through the dense layer.
        
        Mathematical Formula:
        For each sample i and neuron j:
        output[i][j] = Σ(inputs[i][k] × weights[k][j]) + biases[j]
        
        This is a linear transformation of the input, which will later be
        passed through an activation function to introduce non-linearity.
        
        Biological Inspiration:
        - Weights represent synaptic strengths between neurons
        - Biases represent the neuron's baseline activation threshold
        - The sum represents integration of signals from connected neurons
        
        Computational Steps:
        1. Matrix multiplication: inputs × weights
        2. Add bias vector to each row
        3. Store result for next layer
        
        Args:
            inputs: Input matrix [batch_size, n_input_features]
            
        Returns:
            Output matrix [batch_size, n_neurons]
            
        Raises:
            ValueError: If input dimensions don't match layer expectations
        """
        # Validate input dimensions
        if not inputs or len(inputs[0]) != self.n_input_features:
            raise ValueError(
                f"Input features ({len(inputs[0]) if inputs else 0}) "
                f"don't match layer input size ({self.n_input_features})"
            )
        
        # Perform linear transformation: inputs × weights + biases
        self.output = self.matrix_ops.matrix_multiply(inputs, self.weights, self.biases)
        
        return self.output
    
    def get_parameters(self) -> tuple:
        """
        Get the learnable parameters of this layer.
        
        Returns:
            Tuple of (weights, biases) for potential use in training
        """
        return self.weights, self.biases


class LossAndMetrics:
    """
    Loss functions and evaluation metrics for neural network training.
    
    Loss functions measure how far the model's predictions are from the true labels.
    They provide the objective that the model tries to minimize during training.
    
    Metrics provide human-interpretable measures of model performance.
    """
    
    def __init__(self):
        """Initialize the loss calculator with matrix operations helper."""
        self.matrix_ops = NeuralMatrix()

    def categorical_crossentropy_loss(self, predictions: List[List[float]], true_labels: List[int]) -> float:
        """
        Calculate categorical cross-entropy loss for multi-class classification.
        
        Mathematical Formula:
        Loss = -Σ log(predicted_probability_of_true_class) / N
        
        For each sample i with true class yi:
        loss_i = -log(predictions[i][yi])
        
        Total Loss = (1/N) × Σ loss_i
        
        Mathematical Intuition:
        - Cross-entropy measures the "surprise" when predicting with our model
        - If we predict the correct class with high probability: low loss
        - If we predict the correct class with low probability: high loss
        - Logarithm heavily penalizes confident wrong predictions
        
        Why use negative log-likelihood?
        1. log(p) where p ∈ (0,1) gives values in (-∞, 0]
        2. We want loss to be positive, so we negate: -log(p) ∈ [0, ∞)
        3. log(1) = 0, so perfect prediction gives loss = 0
        4. log(p) → -∞ as p → 0, so wrong confident predictions get high loss
        
        Relationship to Information Theory:
        Cross-entropy measures the average number of bits needed to encode events
        from true distribution using predicted distribution.
        
        Args:
            predictions: Model output probabilities [batch_size, num_classes]
                        Each row should be a probability distribution (sum to 1)
            true_labels: True class indices [batch_size]
                        Integer labels indicating correct class for each sample
            
        Returns:
            Average loss across all samples (scalar value)
            
        Example:
            predictions = [[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]
            true_labels = [1, 2]  # True classes are index 1 and 2
            
            Sample 1: predicted prob for class 1 = 0.8, loss = -log(0.8) = 0.22
            Sample 2: predicted prob for class 2 = 0.4, loss = -log(0.4) = 0.92
            Average loss = (0.22 + 0.92) / 2 = 0.57
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Number of predictions must match number of labels")
        
        # Extract predicted probabilities for true classes
        true_class_probabilities = []
        for i in range(len(predictions)):
            true_class_idx = true_labels[i]
            predicted_prob = predictions[i][true_class_idx]
            true_class_probabilities.append(predicted_prob)
        
        # Clip probabilities to prevent log(0) which is undefined
        # log(1e-7) ≈ -16.1, reasonable minimum for numerical stability
        clipped_probabilities = self.matrix_ops.clip_values(true_class_probabilities)
        
        # Calculate negative log-likelihood for each sample
        sample_losses = [-math.log(prob) for prob in clipped_probabilities]
        
        # Return average loss across all samples
        average_loss = sum(sample_losses) / len(sample_losses)
        return average_loss

    def categorical_crossentropy_loss_one_hot(self, predictions: List[List[float]], true_labels_one_hot: List[List[int]]) -> float:
        """
        Calculate categorical cross-entropy loss with one-hot encoded labels.
        
        Mathematical Formula:
        Loss = -Σ Σ true_labels[i][j] × log(predictions[i][j]) / N
        
        This version handles one-hot encoded labels like:
        true_labels = [[0, 1, 0], [0, 0, 1]]  # Instead of [1, 2]
        
        Args:
            predictions: Model output probabilities [batch_size, num_classes]
            true_labels_one_hot: One-hot encoded true labels [batch_size, num_classes]
            
        Returns:
            Average loss across all samples
        """
        total_loss = 0
        num_samples = len(predictions)
        
        for i in range(num_samples):
            sample_loss = 0
            for j in range(len(predictions[i])):
                if true_labels_one_hot[i][j] == 1:  # This is the correct class
                    clipped_prob = max(1e-7, min(predictions[i][j], 1 - 1e-7))
                    sample_loss += -math.log(clipped_prob)
            total_loss += sample_loss
        
        return total_loss / num_samples

    def accuracy(self, predictions: List[List[float]], true_labels: List[int]) -> float:
        """
        Calculate classification accuracy.
        
        Mathematical Formula:
        Accuracy = (Number of Correct Predictions) / (Total Predictions)
        
        For multi-class classification:
        - Predicted class = argmax(prediction probabilities)
        - Correct if predicted class == true class
        
        Args:
            predictions: Model output probabilities [batch_size, num_classes]
            true_labels: True class indices [batch_size]
            
        Returns:
            Accuracy as a fraction between 0.0 and 1.0
            
        Example:
            predictions = [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
            true_labels = [1, 0]
            
            Predicted classes: [1, 0] (argmax of each row)
            True classes:      [1, 0]
            Accuracy: 2/2 = 1.0 (100% correct)
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Number of predictions must match number of labels")
        
        # Get predicted class for each sample (index of maximum probability)
        predicted_classes = self.matrix_ops.argmax_rows(predictions)
        
        # Count correct predictions
        correct_predictions = 0
        for i in range(len(predicted_classes)):
            if predicted_classes[i] == true_labels[i]:
                correct_predictions += 1
        
        # Calculate accuracy as fraction of correct predictions
        accuracy = correct_predictions / len(true_labels)
        return accuracy
    
    def top_k_accuracy(self, predictions: List[List[float]], true_labels: List[int], k: int = 5) -> float:
        """
        Calculate top-k accuracy (prediction is correct if true label is in top k predictions).
        
        Args:
            predictions: Model output probabilities
            true_labels: True class indices  
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy as a fraction between 0.0 and 1.0
        """
        correct = 0
        for i in range(len(predictions)):
            # Get indices of top k predictions
            sorted_indices = sorted(range(len(predictions[i])), 
                                  key=lambda x: predictions[i][x], reverse=True)
            top_k_indices = sorted_indices[:k]
            
            if true_labels[i] in top_k_indices:
                correct += 1
        
        return correct / len(true_labels)


def demonstrate_neural_network():
    """
    Comprehensive demonstration of neural network forward propagation.
    
    This function creates and runs a complete neural network to classify
    spiral-shaped data into 3 classes. It demonstrates all the mathematical
    concepts and operations involved in neural network computation.
    
    Network Architecture:
    Input Layer (2 features) → Hidden Layer (3 neurons, ReLU) → Output Layer (3 classes, Softmax)
    
    Mathematical Flow:
    1. Input: X[300, 2] - 300 samples with 2 features each (x, y coordinates)
    2. Hidden Layer: Linear(2→3) + ReLU activation
       h = ReLU(X × W1 + b1)
    3. Output Layer: Linear(3→3) + Softmax activation  
       output = Softmax(h × W2 + b2)
    4. Loss: Categorical Cross-Entropy
    5. Accuracy: Classification accuracy
    """
    print("=" * 70)
    print("NEURAL NETWORK FROM SCRATCH - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    
    # Step 1: Create synthetic spiral dataset
    print("\n1. DATA GENERATION")
    print("-" * 50)
    print("Creating spiral dataset for 3-class classification...")
    print("Mathematical Approach: Generate points in spiral patterns using:")
    print("  x = r × sin(θ), y = r × cos(θ)")
    print("  where r increases linearly and θ varies with class offset")
    
    # Generate 100 samples per class (300 total)
    X, y = create_data(100, 3)
    
    print(f"Dataset created:")
    print(f"  - Input features (X): {len(X)} samples × {len(X[0])} features")
    print(f"  - Target labels (y): {len(y)} labels (classes 0, 1, 2)")
    print(f"  - Sample input: X[0] = {X[0]}")
    print(f"  - Sample label: y[0] = {y[0]}")
    
    # Step 2: Initialize neural network components
    print("\n2. NEURAL NETWORK INITIALIZATION")
    print("-" * 50)
    
    # Initialize activation functions
    activation_functions = ActivationFunctions()
    
    # Initialize loss and metrics calculator
    loss_calculator = LossAndMetrics()
    
    # Step 3: Create network layers
    print("Creating network architecture:")
    print("  Input → Hidden Layer (2→3 neurons) → Output Layer (3→3 neurons)")
    
    # Hidden layer: 2 input features → 3 neurons
    hidden_layer = DenseLayer(n_input_features=2, n_neurons=3)
    print(f"\nHidden Layer initialized:")
    print(f"  - Weight matrix shape: {len(hidden_layer.weights)} × {len(hidden_layer.weights[0])}")
    print(f"  - Bias vector length: {len(hidden_layer.biases)}")
    print(f"  - Sample weights: {hidden_layer.weights[0][:2]}...")  # Show first few weights
    
    # Output layer: 3 hidden neurons → 3 output classes
    output_layer = DenseLayer(n_input_features=3, n_neurons=3)
    print(f"\nOutput Layer initialized:")
    print(f"  - Weight matrix shape: {len(output_layer.weights)} × {len(output_layer.weights[0])}")
    print(f"  - Bias vector length: {len(output_layer.biases)}")
    
    # Step 4: Forward propagation through hidden layer
    print("\n3. FORWARD PROPAGATION - HIDDEN LAYER")
    print("-" * 50)
    print("Mathematical Operation: h_linear = X × W1 + b1")
    print("  Where:")
    print(f"    X: input matrix [{len(X)} × {len(X[0])}]")
    print(f"    W1: weight matrix [{len(hidden_layer.weights)} × {len(hidden_layer.weights[0])}]")
    print(f"    b1: bias vector [{len(hidden_layer.biases)}]")
    
    # Linear transformation in hidden layer
    hidden_linear_output = hidden_layer.forward(X)
    print(f"\nHidden layer linear output shape: {len(hidden_linear_output)} × {len(hidden_linear_output[0])}")
    print(f"Sample hidden linear output: {hidden_linear_output[0]}")
    
    # Apply ReLU activation
    print("\nApplying ReLU activation: h = max(0, h_linear)")
    print("ReLU Purpose:")
    print("  - Introduces non-linearity (enables learning complex patterns)")
    print("  - Sparse activation (many neurons output 0)")
    print("  - Computationally efficient")
    
    hidden_activated = activation_functions.relu(hidden_linear_output)
    print(f"Sample hidden activated output: {hidden_activated[0]}")
    
    # Count number of activated neurons (non-zero outputs)
    total_activations = sum(1 for row in hidden_activated for val in row if val > 0)
    total_neurons = len(hidden_activated) * len(hidden_activated[0])
    sparsity = (total_neurons - total_activations) / total_neurons * 100
    print(f"ReLU Sparsity: {sparsity:.1f}% of neurons output zero")
    
    # Step 5: Forward propagation through output layer
    print("\n4. FORWARD PROPAGATION - OUTPUT LAYER")
    print("-" * 50)
    print("Mathematical Operation: output_linear = h × W2 + b2")
    print("  Where:")
    print(f"    h: hidden layer output [{len(hidden_activated)} × {len(hidden_activated[0])}]")
    print(f"    W2: output weight matrix [{len(output_layer.weights)} × {len(output_layer.weights[0])}]")
    print(f"    b2: output bias vector [{len(output_layer.biases)}]")
    
    # Linear transformation in output layer
    output_linear = output_layer.forward(hidden_activated)
    print(f"\nOutput layer linear output shape: {len(output_linear)} × {len(output_linear[0])}")
    print(f"Sample output linear (logits): {output_linear[0]}")
    
    # Apply Softmax activation
    print("\nApplying Softmax activation: P(class) = exp(logit) / Σexp(logits)")
    print("Softmax Purpose:")
    print("  - Converts logits to probability distribution")
    print("  - All outputs sum to 1.0")
    print("  - Larger logits get higher probabilities")
    print("  - Differentiable (good for gradient-based optimization)")
    
    final_output = activation_functions.softmax(output_linear)
    print(f"Sample softmax output (probabilities): {final_output[0]}")
    print(f"Probability sum check: {sum(final_output[0]):.6f} (should be 1.0)")
    
    # Step 6: Loss calculation
    print("\n5. LOSS CALCULATION")
    print("-" * 50)
    print("Using Categorical Cross-Entropy Loss:")
    print("Mathematical Formula: L = -Σ log(P(true_class)) / N")
    print("Interpretation:")
    print("  - Measures 'surprise' when making predictions")
    print("  - Perfect prediction (P=1.0): Loss = -log(1) = 0")
    print("  - Wrong prediction (P→0): Loss = -log(0) → ∞")
    print("  - Heavily penalizes confident wrong predictions")
    
    loss = loss_calculator.categorical_crossentropy_loss(final_output, y)
    print(f"\nCalculated Loss: {loss:.4f}")
    
    # Interpret loss value
    print("Loss Interpretation:")
    if loss < 0.5:
        print("  - Excellent: Model predictions are very confident and mostly correct")
    elif loss < 1.0:
        print("  - Good: Model is reasonably confident in correct predictions")  
    elif loss < 1.5:
        print("  - Fair: Model has some confidence but many errors")
    else:
        print("  - Poor: Model predictions are not much better than random guessing")
    
    # Step 7: Accuracy calculation
    print("\n6. ACCURACY EVALUATION")
    print("-" * 50)
    print("Classification Accuracy: (Correct Predictions) / (Total Predictions)")
    print("Process:")
    print("  1. For each sample, find class with highest probability (argmax)")
    print("  2. Compare predicted class with true class")
    print("  3. Calculate percentage of correct predictions")
    
    accuracy = loss_calculator.accuracy(final_output, y)
    print(f"\nCalculated Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Show some example predictions
    print("\nSample Predictions:")
    print("Sample | True Class | Predicted Probabilities          | Predicted Class | Correct?")
    print("-" * 80)
    
    predicted_classes = loss_calculator.matrix_ops.argmax_rows(final_output)
    for i in range(min(5, len(final_output))):  # Show first 5 samples
        probs = [f"{p:.3f}" for p in final_output[i]]
        correct = "✓" if predicted_classes[i] == y[i] else "✗"
        print(f"   {i:2d}   |     {y[i]}      | [{', '.join(probs)}] |      {predicted_classes[i]}       |    {correct}")
    
    # Step 8: Summary and educational insights
    print("\n7. EDUCATIONAL SUMMARY")
    print("-" * 50)
    print("Key Mathematical Concepts Demonstrated:")
    print("✓ Matrix Multiplication: Core operation for linear transformations")
    print("✓ Activation Functions: ReLU (hidden) and Softmax (output)")
    print("✓ Forward Propagation: Information flow through network layers")
    print("✓ Loss Function: Cross-entropy for measuring prediction quality")
    print("✓ Probability Distributions: Softmax creates valid probabilities")
    print("✓ Classification Metrics: Accuracy for model evaluation")
    
    print(f"\nNetwork Performance on {len(X)} samples:")
    print(f"  Loss: {loss:.4f} (lower is better)")
    print(f"  Accuracy: {accuracy*100:.1f}% (higher is better)")
    
    random_baseline = 1.0 / 3  # Random guessing for 3 classes
    print(f"  Random baseline accuracy: {random_baseline*100:.1f}%")
    
    if accuracy > random_baseline:
        improvement = (accuracy - random_baseline) / random_baseline * 100
        print(f"  Improvement over random: +{improvement:.1f}%")
        print("  ✓ Network is learning meaningful patterns!")
    else:
        print("  ⚠ Network performance is not better than random guessing")
        print("    This is expected with random weights (no training)")
    
    print("\nNext Steps for Learning:")
    print("• Implement backpropagation to update weights")
    print("• Add gradient descent optimization")
    print("• Experiment with different architectures")
    print("• Try different activation functions")
    print("• Implement regularization techniques")
    
    print("=" * 70)
    

if __name__ == "__main__":
    """
    Main execution block - demonstrates the complete neural network.
    
    This script showcases a neural network implementation from scratch,
    providing educational value for understanding the mathematics and
    concepts behind deep learning.
    """
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run the comprehensive demonstration
    demonstrate_neural_network()
