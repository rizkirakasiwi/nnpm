"""
Synthetic Dataset Generation for Neural Network Classification

This module creates spiral-shaped datasets for multi-class classification problems.
The data generation uses mathematical functions to create visually separable classes
that are arranged in spiral patterns, providing an excellent test case for neural networks.

Mathematical Foundation:
- Polar coordinates: (r, θ) where r is radius and θ is angle
- Cartesian conversion: x = r × sin(θ), y = r × cos(θ)
- Spiral generation: θ increases with class-specific offsets
- Noise injection: Gaussian noise for realistic data variation

Educational Purpose:
This synthetic dataset helps visualize how neural networks learn to separate
non-linearly separable data, which is impossible with linear classifiers.
"""

import numpy as np

# Set random seed for reproducible dataset generation
# This ensures the same data is generated every time for consistent testing
np.random.seed(0)


def create_data(samples_per_class: int, num_classes: int) -> tuple:
    """
    Generate synthetic spiral dataset for multi-class classification.
    
    Mathematical Approach:
    Creates data points arranged in spiral patterns, with each class forming
    its own spiral arm. This creates a dataset that is:
    - Non-linearly separable (linear classifiers will fail)
    - Visually interpretable (can be plotted in 2D)
    - Challenging enough to demonstrate neural network capabilities
    
    Spiral Generation Mathematics:
    For each class c and point i:
    1. Radius: r[i] = i / samples_per_class (linearly increasing from 0 to 1)
    2. Base angle: θ_base[i] = c × 4 + (i / samples_per_class) × 4
    3. Noise: θ_noise[i] ~ N(0, 0.2²) (Gaussian noise)
    4. Final angle: θ[i] = θ_base[i] + θ_noise[i]
    5. Coordinates: x[i] = r[i] × sin(θ[i] × 2.5), y[i] = r[i] × cos(θ[i] × 2.5)
    
    Why this pattern works well:
    - Each class starts from origin and spirals outward
    - Class separation increases with distance from origin
    - Angular offset (class × 4) creates distinct spiral arms
    - Multiplicative factor (2.5) controls spiral tightness
    - Gaussian noise adds realistic data variation
    
    Mathematical Visualization:
    Class 0: θ ∈ [0, 4] + noise, creates spiral from 0° to ~1440°
    Class 1: θ ∈ [4, 8] + noise, creates spiral from ~1440° to ~2880°
    Class 2: θ ∈ [8, 12] + noise, creates spiral from ~2880° to ~4320°
    
    Args:
        samples_per_class: Number of data points to generate per class
        num_classes: Number of distinct classes (spiral arms)
        
    Returns:
        tuple: (X, y) where:
            X: List of [x, y] coordinate pairs, shape [total_samples, 2]
            y: List of class labels (integers), shape [total_samples]
            total_samples = samples_per_class × num_classes
            
    Example:
        X, y = create_data(100, 3)
        # Creates 300 samples total (100 per class)
        # X contains [(x1,y1), (x2,y2), ...] coordinate pairs
        # y contains [0,0,0,...,1,1,1,...,2,2,2,...] class labels
    """
    # Calculate total number of samples
    total_samples = samples_per_class * num_classes
    
    # Initialize coordinate matrix: [total_samples, 2] for (x, y) coordinates
    # Using float64 for numerical precision in trigonometric calculations
    X = np.zeros((total_samples, 2), dtype=np.float64)
    
    # Initialize label vector: [total_samples] for class indices
    # Using uint8 to save memory (can represent 0-255, sufficient for most classification tasks)
    y = np.zeros(total_samples, dtype=np.uint8)
    
    # Generate data for each class
    for class_index in range(num_classes):
        print(f"Generating class {class_index}...")
        
        # Calculate sample indices for this class
        # Class 0: indices 0 to samples_per_class-1
        # Class 1: indices samples_per_class to 2*samples_per_class-1, etc.
        start_idx = samples_per_class * class_index
        end_idx = samples_per_class * (class_index + 1)
        sample_indices = range(start_idx, end_idx)
        
        # Generate radius values: linearly increasing from 0 to 1
        # This makes points start near origin and spiral outward
        # Mathematical: r[i] = i / (samples_per_class - 1) for i ∈ [0, samples_per_class-1]
        r = np.linspace(0.0, 1.0, samples_per_class)
        
        # Generate base angles with class-specific offset
        # Each class gets 4 radian offset (≈ 229.2 degrees)
        # This separates spiral arms in angular space
        # Mathematical: θ_base = class_index × 4 + (progress × 4)
        #               where progress ∈ [0, 1] represents position along spiral
        angle_start = class_index * 4
        angle_end = (class_index + 1) * 4
        base_angles = np.linspace(angle_start, angle_end, samples_per_class)
        
        # Add Gaussian noise to angles for realistic data variation
        # Standard deviation of 0.2 radians (≈ 11.5 degrees)
        # This prevents perfectly smooth spirals and adds classification challenge
        # Mathematical: θ_noise ~ N(0, 0.2²)
        noise = np.random.normal(0, 0.2, samples_per_class)
        final_angles = base_angles + noise
        
        # Convert from polar to Cartesian coordinates
        # The factor 2.5 controls how tightly the spiral winds
        # Larger values create tighter spirals, smaller values create looser spirals
        # Mathematical transformation:
        #   x = r × sin(θ × spiral_factor)
        #   y = r × cos(θ × spiral_factor)
        spiral_factor = 2.5
        x_coords = r * np.sin(final_angles * spiral_factor)
        y_coords = r * np.cos(final_angles * spiral_factor)
        
        # Store coordinates for this class
        # np.c_ concatenates arrays column-wise: creates [x_coords, y_coords] pairs
        X[sample_indices] = np.c_[x_coords, y_coords]
        
        # Assign class labels
        y[sample_indices] = class_index
        
        # Print some statistics for educational purposes
        print(f"  Class {class_index} statistics:")
        print(f"    Samples: {samples_per_class}")
        print(f"    Radius range: [{r.min():.3f}, {r.max():.3f}]")
        print(f"    Angle range: [{final_angles.min():.3f}, {final_angles.max():.3f}] radians")
        print(f"    X coordinate range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
        print(f"    Y coordinate range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
    
    # Convert numpy arrays to Python lists for compatibility with pure Python neural network
    # This ensures the dataset works with our from-scratch implementation
    X_list = X.tolist()
    y_list = y.tolist()
    
    print(f"\nDataset generation complete:")
    print(f"  Total samples: {len(X_list)}")
    print(f"  Features per sample: {len(X_list[0])}")
    print(f"  Classes: {num_classes} (labels 0 to {num_classes-1})")
    print(f"  Class distribution: {samples_per_class} samples per class")
    
    # Verify data integrity
    assert len(X_list) == len(y_list), "Feature and label counts must match"
    assert all(len(sample) == 2 for sample in X_list), "Each sample must have exactly 2 features"
    assert min(y_list) == 0 and max(y_list) == num_classes - 1, "Labels must be in range [0, num_classes-1]"
    
    return X_list, y_list


def visualize_data_info(X, y):
    """
    Print detailed information about the generated dataset.
    
    This function provides statistical analysis of the dataset for educational purposes,
    helping users understand the characteristics of the spiral data.
    
    Args:
        X: Feature matrix (list of [x, y] coordinate pairs)
        y: Label vector (list of class indices)
    """
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    # Basic statistics
    num_samples = len(X)
    num_features = len(X[0]) if X else 0
    unique_classes = sorted(set(y))
    
    print(f"Dataset Overview:")
    print(f"  Total samples: {num_samples}")
    print(f"  Features per sample: {num_features}")
    print(f"  Number of classes: {len(unique_classes)}")
    print(f"  Class labels: {unique_classes}")
    
    # Class distribution
    print(f"\nClass Distribution:")
    for class_label in unique_classes:
        count = y.count(class_label)
        percentage = (count / num_samples) * 100
        print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
    
    # Feature statistics
    if X:
        x_coords = [sample[0] for sample in X]
        y_coords = [sample[1] for sample in X]
        
        print(f"\nFeature Statistics:")
        print(f"  X-coordinate range: [{min(x_coords):.3f}, {max(x_coords):.3f}]")
        print(f"  Y-coordinate range: [{min(y_coords):.3f}, {max(y_coords):.3f}]")
        print(f"  X-coordinate mean: {sum(x_coords)/len(x_coords):.3f}")
        print(f"  Y-coordinate mean: {sum(y_coords)/len(y_coords):.3f}")
    
    print("="*50)


# Example usage and testing
if __name__ == "__main__":
    """
    Demonstrate the dataset generation with detailed explanations.
    """
    print("Spiral Dataset Generation Demonstration")
    print("="*60)
    
    # Generate a small dataset for demonstration
    print("Generating spiral dataset...")
    X, y = create_data(samples_per_class=50, num_classes=3)
    
    # Show detailed analysis
    visualize_data_info(X, y)
    
    # Show first few samples
    print(f"\nFirst 5 samples:")
    print("Index | X-coord | Y-coord | Class")
    print("-" * 35)
    for i in range(min(5, len(X))):
        print(f"{i:5d} | {X[i][0]:7.3f} | {X[i][1]:7.3f} | {y[i]:5d}")
    
    print(f"\nDataset ready for neural network training!")
    print(f"This spiral dataset provides an excellent test case because:")
    print(f"  • It's non-linearly separable (linear classifiers will fail)")
    print(f"  • It's visually interpretable (can be plotted)")
    print(f"  • It demonstrates neural network's ability to learn complex patterns")
    print(f"  • It has clear class boundaries despite overlapping regions")
