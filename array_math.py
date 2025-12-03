import numpy as np

def add_arrays(arr1, arr2):
    """Add two arrays element-wise.

    Parameters:
    arr1 (array-like): First input array.
    arr2 (array-like): Second input array.

    Returns:
    numpy.ndarray: Element-wise sum of arr1 and arr2.
    """
    return np.add(arr1, arr2)

def multiply_arrays(arr1, arr2):
    """Multiply two arrays element-wise.

    Parameters:
    arr1 (array-like): First input array.
    arr2 (array-like): Second input array.

    Returns:
    numpy.ndarray: Element-wise product of arr1 and arr2.
    """
    return np.multiply(arr1, arr2)

def dot_product(arr1, arr2):
    """Compute the dot product of two arrays.

    Parameters:
    arr1 (array-like): First input array.
    arr2 (array-like): Second input array.

    Returns:
    numpy.ndarray: Dot product of arr1 and arr2.
    """
    return np.dot(arr1, arr2)

def transpose_array(arr):
    """Transpose the given array.

    Parameters:
    arr (array-like): Input array.

    Returns:
    numpy.ndarray: Transposed array.
    """
    return np.transpose(arr)

def reshape_array(arr, new_shape):
    """Reshape the given array to a new shape.

    Parameters:
    arr (array-like): Input array.
    new_shape (tuple): Desired shape.

    Returns:
    numpy.ndarray: Reshaped array.
    """
    return np.reshape(arr, new_shape)

def array_mean(arr):
    """Calculate the mean of the array elements.

    Parameters:
    arr (array-like): Input array.

    Returns:
    float: Mean of the array elements.
    """
    return np.mean(arr)

def array_std(arr):
    """Calculate the standard deviation of the array elements.

    Parameters:
    arr (array-like): Input array.

    Returns:
    float: Standard deviation of the array elements.
    """
    return np.std(arr)

def array_sum(arr):
    """Calculate the sum of the array elements.

    Parameters:
    arr (array-like): Input array.

    Returns:
    float: Sum of the array elements.
    """
    return np.sum(arr)

def array_max(arr):
    """Find the maximum value in the array.

    Parameters:
    arr (array-like): Input array.

    Returns:
    float: Maximum value in the array.
    """
    return np.max(arr)

def array_min(arr):
    """Find the minimum value in the array.

    Parameters:
    arr (array-like): Input array.

    Returns:
    float: Minimum value in the array.
    """
    return np.min(arr)

def flatten_array(arr):
    """Flatten the given array to 1D.

    Parameters:
    arr (array-like): Input array.

    Returns:
    numpy.ndarray: Flattened 1D array.
    """
    return np.ravel(arr)

def concatenate_arrays(arr1, arr2, axis=0):
    """Concatenate two arrays along a specified axis.

    Parameters:
    arr1 (array-like): First input array.
    arr2 (array-like): Second input array.
    axis (int): Axis along which the arrays will be joined.

    Returns:
    numpy.ndarray: Concatenated array.
    """
    return np.concatenate((arr1, arr2), axis=axis)

def split_array(arr, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.

    Parameters:
    arr (array-like): Input array.
    indices_or_sections (int or 1-D array): If an integer, it specifies the number of equal arrays to return. 
                                            If a 1-D array, it specifies the indices at which to split.
    axis (int): Axis along which to split the array.

    Returns:
    list of numpy.ndarray: List of sub-arrays.
    """
    return np.split(arr, indices_or_sections, axis=axis)

def unique_elements(arr):
    """Find the unique elements of the array.

    Parameters:
    arr (array-like): Input array.

    Returns:
    numpy.ndarray: Array of unique elements.
    """
    return np.unique(arr)

def sort_array(arr):
    """Sort the array elements in ascending order.

    Parameters:
    arr (array-like): Input array.

    Returns:
    numpy.ndarray: Sorted array.
    """
    return np.sort(arr)

def clip_array(arr, min_value, max_value):
    """Clip the array elements to be within the specified range.

    Parameters:
    arr (array-like): Input array.
    min_value (float): Minimum value.
    max_value (float): Maximum value.

    Returns:
    numpy.ndarray: Clipped array.
    """
    return np.clip(arr, min_value, max_value)

def power_array(arr, exponent):
    """Raise each element of the array to the specified power.

    Parameters:
    arr (array-like): Input array.
    exponent (float): Exponent to which each element is raised.

    Returns:
    numpy.ndarray: Array with elements raised to the specified power.
    """
    return np.power(arr, exponent)

def sqrt_array(arr):
    """Calculate the square root of each element in the array.

    Parameters:
    arr (array-like): Input array.

    Returns:
    numpy.ndarray: Array with square roots of the original elements.
    """
    return np.sqrt(arr)

def log_array(arr):
    """Calculate the natural logarithm of each element in the array.

    Parameters:
    arr (array-like): Input array.

    Returns:
    numpy.ndarray: Array with natural logarithms of the original elements.
    """
    return np.log(arr)

def exp_array(arr):
    """Calculate the exponential of each element in the array.

    Parameters:
    arr (array-like): Input array.

    Returns:
    numpy.ndarray: Array with exponentials of the original elements.
    """
    return np.exp(arr)

def round_array(arr, decimals=0):
    """Round each element in the array to the specified number of decimals.

    Parameters:
    arr (array-like): Input array.
    decimals (int): Number of decimal places to round to.

    Returns:
    numpy.ndarray: Array with rounded elements.
    """
    return np.round(arr, decimals=decimals)

def histogram_array(arr, bins=10, range=None):
    """Compute the histogram of the array elements.

    Parameters:
    arr (array-like): Input array.
    bins (int or sequence): Number of bins or the bin edges.
    range (tuple): The lower and upper range of the bins.

    Returns:
    tuple: Histogram values and bin edges.
    """
    return np.histogram(arr, bins=bins, range=range)

def covariance_matrix(arr1, arr2):
    """Compute the covariance matrix of two arrays.

    Parameters:
    arr1 (array-like): First input array.
    arr2 (array-like): Second input array.

    Returns:
    numpy.ndarray: Covariance matrix of arr1 and arr2.
    """
    return np.cov(arr1, arr2)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================
# This section demonstrates all the functions defined above with examples.

if __name__ == "__main__":
    print("=" * 70)
    print("NumPy Array Math Operations Demonstration")
    print("=" * 70)
    
    # Create sample arrays for demonstration
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([5, 4, 3, 2, 1])
    arr2d_1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2d_2 = np.array([[7, 8], [9, 10], [11, 12]])
    
    print("\n" + "-" * 70)
    print("Basic Array Operations")
    print("-" * 70)
    
    # Addition
    print(f"\nArray 1: {arr1}")
    print(f"Array 2: {arr2}")
    print(f"add_arrays(arr1, arr2): {add_arrays(arr1, arr2)}")
    print("  → Adds corresponding elements: [1+5, 2+4, 3+3, 4+2, 5+1]")
    
    # Multiplication
    print(f"\nmultiply_arrays(arr1, arr2): {multiply_arrays(arr1, arr2)}")
    print("  → Multiplies corresponding elements element-wise")
    
    # Dot product
    print(f"\ndot_product(arr1, arr2): {dot_product(arr1, arr2)}")
    print("  → Computes dot product: 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 35")
    
    print("\n" + "-" * 70)
    print("Array Shape Operations")
    print("-" * 70)
    
    # Transpose
    print(f"\n2D Array 1:\n{arr2d_1}")
    print(f"transpose_array(arr2d_1):\n{transpose_array(arr2d_1)}")
    print("  → Rows become columns and columns become rows")
    
    # Reshape
    print(f"\nArray 1: {arr1}")
    reshaped = reshape_array(arr1, (5, 1))
    print(f"reshape_array(arr1, (5, 1)):\n{reshaped}")
    print("  → Reorganizes 5 elements into 5 rows, 1 column")
    
    # Flatten
    print(f"\n2D Array 1:\n{arr2d_1}")
    flattened = flatten_array(arr2d_1)
    print(f"flatten_array(arr2d_1): {flattened}")
    print("  → Converts 2D array to 1D array")
    
    print("\n" + "-" * 70)
    print("Statistical Operations")
    print("-" * 70)
    
    # Mean
    print(f"\nArray 1: {arr1}")
    print(f"array_mean(arr1): {array_mean(arr1)}")
    print("  → Average: (1+2+3+4+5)/5 = 3.0")
    
    # Standard deviation
    print(f"\narray_std(arr1): {array_std(arr1)}")
    print("  → Measures how spread out the values are")
    
    # Sum
    print(f"\narray_sum(arr1): {array_sum(arr1)}")
    print("  → Total: 1+2+3+4+5 = 15")
    
    # Max and Min
    print(f"\narray_max(arr1): {array_max(arr1)}")
    print(f"array_min(arr1): {array_min(arr1)}")
    print("  → Find maximum and minimum values")
    
    print("\n" + "-" * 70)
    print("Array Combination Operations")
    print("-" * 70)
    
    # Concatenate
    arr_a = np.array([1, 2, 3])
    arr_b = np.array([4, 5, 6])
    print(f"\nArray A: {arr_a}")
    print(f"Array B: {arr_b}")
    print(f"concatenate_arrays(arr_a, arr_b): {concatenate_arrays(arr_a, arr_b)}")
    print("  → Joins arrays end-to-end")
    
    # Split
    arr_split = np.array([1, 2, 3, 4, 5, 6])
    print(f"\nArray to split: {arr_split}")
    split_result = split_array(arr_split, 3)
    print(f"split_array(arr, 3): {split_result}")
    print("  → Divides array into 3 equal parts")
    
    # Unique elements
    arr_with_duplicates = np.array([1, 2, 2, 3, 3, 3, 4])
    print(f"\nArray with duplicates: {arr_with_duplicates}")
    print(f"unique_elements(arr): {unique_elements(arr_with_duplicates)}")
    print("  → Finds unique values: each element appears once")
    
    print("\n" + "-" * 70)
    print("Sorting and Ordering")
    print("-" * 70)
    
    # Sort
    unsorted_arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"\nUnsorted array: {unsorted_arr}")
    print(f"sort_array(arr): {sort_array(unsorted_arr)}")
    print("  → Arranges elements in ascending order")
    
    print("\n" + "-" * 70)
    print("Element-wise Mathematical Operations")
    print("-" * 70)
    
    arr_math = np.array([1, 2, 3, 4])
    
    # Power
    print(f"\nArray: {arr_math}")
    print(f"power_array(arr, 2): {power_array(arr_math, 2)}")
    print("  → Raises each element to power of 2: [1^2, 2^2, 3^2, 4^2]")
    
    # Square root
    print(f"\nsqrt_array(arr): {sqrt_array(arr_math)}")
    print("  → Computes square root of each element")
    
    # Exponential
    arr_small = np.array([0, 1, 2])
    print(f"\nArray: {arr_small}")
    print(f"exp_array(arr): {exp_array(arr_small)}")
    print("  → Computes e^x for each element")
    
    # Natural logarithm (add 1 to avoid log(0))
    print(f"\nlog_array(arr+1): {log_array(arr_small + 1)}")
    print("  → Computes natural logarithm of each element")
    
    # Clip
    arr_to_clip = np.array([1, 2, 3, 4, 5, 6])
    print(f"\nArray: {arr_to_clip}")
    print(f"clip_array(arr, 2, 5): {clip_array(arr_to_clip, 2, 5)}")
    print("  → Limits values to range [2, 5]")
    
    # Round
    arr_float = np.array([1.234, 2.567, 3.891])
    print(f"\nFloat array: {arr_float}")
    print(f"round_array(arr, 2): {round_array(arr_float, 2)}")
    print("  → Rounds to 2 decimal places")
    
    print("\n" + "-" * 70)
    print("Statistical Distribution")
    print("-" * 70)
    
    # Histogram
    hist_data = np.array([1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6])
    hist_values, bin_edges = histogram_array(hist_data, bins=3)
    print(f"\nData: {hist_data}")
    print(f"histogram_array(data, bins=3):")
    print(f"  Values (counts): {hist_values}")
    print(f"  Bin edges: {bin_edges}")
    print("  → Shows frequency of values in different ranges")
    
    print("\n" + "-" * 70)
    print("Covariance Analysis")
    print("-" * 70)
    
    # Covariance
    var1 = np.array([1, 2, 3, 4, 5])
    var2 = np.array([2, 4, 5, 4, 6])
    cov = covariance_matrix(var1, var2)
    print(f"\nVariable 1: {var1}")
    print(f"Variable 2: {var2}")
    print(f"covariance_matrix(var1, var2):\n{cov}")
    print("  → Shows how two variables vary together")
    print("    Diagonal: variance of each variable")
    print("    Off-diagonal: covariance between variables")
    
    print("\n" + "-" * 70)
    print("Matrix Operations Example")
    print("-" * 70)
    
    # Matrix multiplication example
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    print(f"\nMatrix 1:\n{mat1}")
    print(f"Matrix 2:\n{mat2}")
    print(f"dot_product(mat1, mat2):\n{dot_product(mat1, mat2)}")
    print("  → Matrix multiplication of 2×2 matrices")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)

