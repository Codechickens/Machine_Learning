"""
NumPy Array Tutorial
====================
This tutorial covers the fundamentals of creating and working with NumPy arrays.
NumPy is the fundamental package for numerical computing in Python.

Key Benefits of NumPy Arrays:
- Much faster than Python lists (optimized C code)
- Less memory consumption
- Broadcasting capabilities for element-wise operations
- Convenient for mathematical and scientific computing
"""

import numpy as np

# Display the version of NumPy being used
print(f"NumPy Version: {np.__version__}\n")

# ============================================================================
# 1. CREATING ARRAYS
# ============================================================================
# Arrays are the core data structure in NumPy. They are similar to Python lists
# but more efficient and with more functionality for numerical operations.

print("=" * 70)
print("1. CREATING ARRAYS")
print("=" * 70)

# 1.1 Create array from Python list
# np.array() converts a Python list into a NumPy array
# This is the most straightforward way to create arrays from existing data
arr1 = np.array([1, 2, 3, 4, 5])
print("\nFrom Python list:")
print(f"arr1 = {arr1}")
print(f"Type: {type(arr1)}")  # Shows the type is numpy.ndarray (n-dimensional array)
print(f"Shape: {arr1.shape}")  # Shape (5,) means 1D array with 5 elements
print(f"Data type: {arr1.dtype}")  # dtype shows the data type of elements

# 1.2 Create 2D array (matrix)
# Nested lists create multi-dimensional arrays
# Each inner list becomes a row in the 2D array
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:")
print(f"arr2d =\n{arr2d}")
print(f"Shape: {arr2d.shape}")  # Shape (3, 3) means 3 rows and 3 columns

# 1.3 Create 3D array (tensor)
# Further nesting creates higher-dimensional arrays
# Useful for representing batches of images, video frames, etc.
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3D Array:")
print(f"arr3d =\n{arr3d}")
print(f"Shape: {arr3d.shape}")  # Shape (2, 2, 2) means 2x2x2 array

# 1.4 Arrays with specific data types
# The dtype parameter specifies the data type of array elements
# Common dtypes: int, float, complex, bool, etc.
# Specifying dtype can save memory and ensure correct computations
arr_int = np.array([1, 2, 3], dtype=int)  # Integers (typically 64-bit)
arr_float = np.array([1, 2, 3], dtype=float)  # Floating point numbers
arr_complex = np.array([1, 2, 3], dtype=complex)  # Complex numbers
print(f"\nInteger array: {arr_int} (dtype: {arr_int.dtype})")
print(f"Float array: {arr_float} (dtype: {arr_float.dtype})")
print(f"Complex array: {arr_complex} (dtype: {arr_complex.dtype})")

# ============================================================================
# 2. CREATE ARRAYS WITH FUNCTIONS
# ============================================================================
# NumPy provides convenient functions to create arrays with specific patterns
# without manually typing all values. This is very useful for large arrays.

print("\n" + "=" * 70)
print("2. CREATE ARRAYS WITH FUNCTIONS")
print("=" * 70)

# 2.1 zeros - create array filled with zeros
# Useful for initializing arrays before filling with computed values
zeros = np.zeros(5)  # Create 1D array of 5 zeros
print(f"\nnp.zeros(5) = {zeros}")

zeros_2d = np.zeros((3, 4))  # Create 2D array (3 rows, 4 columns) of zeros
print(f"np.zeros((3, 4)) =\n{zeros_2d}")

# 2.2 ones - create array filled with ones
# Often used for masking operations or as identity-like initializations
ones = np.ones(5)  # Create 1D array of 5 ones
print(f"\nnp.ones(5) = {ones}")

ones_2d = np.ones((2, 3))  # Create 2D array (2 rows, 3 columns) of ones
print(f"np.ones((2, 3)) =\n{ones_2d}")

# 2.3 full - create array filled with a specific value
# More flexible than zeros/ones - can use any value
full = np.full(5, 7)  # Create 1D array of 5 sevens
print(f"\nnp.full(5, 7) = {full}")

# 2.4 arange - create array with evenly spaced values (like Python's range)
# arange(start, stop, step) - similar to range() but returns an array
# stop value is NOT included (similar to range())
arange_arr = np.arange(0, 10, 2)  # Values: 0, 2, 4, 6, 8
print(f"\nnp.arange(0, 10, 2) = {arange_arr}")

# 2.5 linspace - create array with specified number of evenly spaced elements
# linspace(start, stop, num_elements) - generates num_elements between start and stop
# INCLUDES both start and stop values (unlike arange)
linspace_arr = np.linspace(0, 10, 5)  # 5 evenly spaced values from 0 to 10
print(f"np.linspace(0, 10, 5) = {linspace_arr}")

# 2.6 eye - create identity matrix
# Identity matrix has 1s on diagonal and 0s elsewhere
# Useful for matrix multiplication and linear algebra operations
eye = np.eye(3)  # Create 3x3 identity matrix
print(f"\nnp.eye(3) =\n{eye}")

# 2.7 Random arrays
# np.random.rand() - uniform distribution between 0 and 1
# Useful for initializing neural network weights, generating test data
random_arr = np.random.rand(5)  # 5 random floats between 0 and 1
print(f"\nnp.random.rand(5) = {random_arr}")

# np.random.randint() - random integers in specified range
# Useful for generating random indices or discrete values
random_arr_2d = np.random.randint(0, 10, (3, 3))  # 3x3 array of random integers 0-9
print(f"np.random.randint(0, 10, (3, 3)) =\n{random_arr_2d}")

# np.random.randn() - standard normal distribution (mean=0, std=1)
# Useful for initializing weights in neural networks
random_normal = np.random.randn(5)  # 5 random values from standard normal
print(f"np.random.randn(5) = {random_normal}")

# ============================================================================
# 3. ARRAY ATTRIBUTES
# ============================================================================
# Array attributes provide metadata about the array structure and data type.
# Understanding these is crucial for debugging shape-related errors.

print("\n" + "=" * 70)
print("3. ARRAY ATTRIBUTES")
print("=" * 70)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nArray:\n{arr}")
print(f"Shape: {arr.shape}")           # (3, 3) - number of elements in each dimension
print(f"Size: {arr.size}")             # 9 - total number of elements (product of dimensions)
print(f"Ndim: {arr.ndim}")             # 2 - number of dimensions
print(f"Dtype: {arr.dtype}")           # int64 or similar - data type of elements
print(f"Itemsize: {arr.itemsize}")     # 8 - bytes used by each element
print(f"Nbytes: {arr.nbytes}")         # 72 - total bytes used by entire array
print(f"Strides: {arr.strides}")       # (24, 8) - bytes to step in each dimension

# ============================================================================
# 4. INDEXING AND SLICING
# ============================================================================
# Indexing retrieves individual elements, while slicing retrieves ranges.
# This is fundamental for accessing and manipulating array data.

print("\n" + "=" * 70)
print("4. INDEXING AND SLICING")
print("=" * 70)

arr = np.arange(10)  # Creates array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"\nArray: {arr}")

# 4.1 Single element indexing
# Use square brackets with index (0-based indexing)
# Negative indices count from the end: -1 is last, -2 is second-to-last
print(f"arr[0] = {arr[0]}")      # First element
print(f"arr[5] = {arr[5]}")      # Element at index 5
print(f"arr[-1] = {arr[-1]}")    # Last element (index -1)
print(f"arr[-2] = {arr[-2]}")    # Second to last element

# 4.2 Slicing
# Syntax: arr[start:stop:step]
# start - inclusive, stop - exclusive, step - increment
print(f"\narr[2:5] = {arr[2:5]}")      # Elements from index 2 to 4 (5 is excluded)
print(f"arr[:5] = {arr[:5]}")          # First 5 elements (start defaults to 0)
print(f"arr[5:] = {arr[5:]}")          # From index 5 to end (stop defaults to end)
print(f"arr[::2] = {arr[::2]}")        # Every 2nd element (start:stop defaults to full range)
print(f"arr[::-1] = {arr[::-1]}")      # Reverse array (step=-1)

# 4.3 2D array indexing
# For 2D arrays, use [row, column] or [row][column]
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D Array:\n{arr2d}")
print(f"arr2d[0] = {arr2d[0]}")        # First row
print(f"arr2d[1, 2] = {arr2d[1, 2]}")  # Element at row 1, column 2 (value 6)
print(f"arr2d[0:2, 1:3] =\n{arr2d[0:2, 1:3]}")  # Rows 0-1, columns 1-2 (subarray)

# 4.4 Boolean indexing
# Use a boolean array to select elements meeting a condition
# Returns only elements where condition is True
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3  # Creates boolean array [False, False, False, True, True, True]
print(f"\nArray: {arr}")
print(f"arr > 3: {mask}")              # Boolean condition
print(f"arr[arr > 3] = {arr[arr > 3]}")  # Selects elements where condition is True

# 4.5 Fancy indexing
# Use an array of indices to select elements
# Can select arbitrary elements in any order
arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]  # Select elements at indices 0, 2, and 4
print(f"\nArray: {arr}")
print(f"arr[[0, 2, 4]] = {arr[[0, 2, 4]]}")  # Results in [10, 30, 50]

# ============================================================================
# 5. RESHAPING ARRAYS
# ============================================================================
# Reshaping changes the dimensions of an array without changing its data.
# The total number of elements must remain the same after reshaping.

print("\n" + "=" * 70)
print("5. RESHAPING ARRAYS")
print("=" * 70)

# 5.1 reshape - change shape while keeping data
# reshape(new_shape) reorganizes elements into new dimensions
# Must have same total number of elements: 12 elements can be reshaped to (3,4), (2,6), (4,3), etc.
arr = np.arange(12)  # Creates [0, 1, 2, ..., 11]
print(f"\nOriginal array: {arr}")
reshaped = arr.reshape(3, 4)  # 12 elements → 3 rows × 4 columns
print(f"reshaped to (3, 4):\n{reshaped}")

reshaped_3d = arr.reshape(2, 2, 3)  # 12 elements → 2×2×3 cube
print(f"reshaped to (2, 2, 3):\n{reshaped_3d}")

# 5.2 flatten - convert any array to 1D
# flatten() creates a new array with all elements in a single dimension
# Always creates a copy (not a view)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
flattened = arr2d.flatten()  # Converts to 1D: [1, 2, 3, 4, 5, 6]
print(f"\n2D array:\n{arr2d}")
print(f"Flattened: {flattened}")

# 5.3 ravel - convert to 1D (returns view if possible)
# ravel() is similar to flatten() but returns a view when possible (more efficient)
# A view shares the same data, so changes affect the original
raveled = arr2d.ravel()
print(f"Raveled: {raveled}")

# 5.4 transpose - flip rows and columns
# For 2D arrays: rows become columns and columns become rows
# For nD arrays: reverses all dimensions
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr2d.T  # T is shorthand for transpose()
print(f"\nOriginal:\n{arr2d}")
print(f"Transposed:\n{transposed}")

# 5.5 expand_dims - add a new dimension
# Useful for adding batch dimension or preparing for broadcasting
arr1d = np.array([1, 2, 3])
expanded = np.expand_dims(arr1d, axis=0)  # Add dimension at axis 0
print(f"\n1D array: {arr1d}")
print(f"Expanded at axis 0 (shape {expanded.shape}):\n{expanded}")

# 5.6 squeeze - remove dimensions of size 1
# Removes all axes with length 1, useful after aggregations
arr = np.array([[[1], [2], [3]]])  # Shape (1, 3, 1)
squeezed = np.squeeze(arr)  # Removes dimensions with size 1
print(f"\nOriginal shape: {arr.shape}")
print(f"Squeezed shape: {squeezed.shape}")
print(f"Result: {squeezed}")

# ============================================================================
# 6. ARRAY OPERATIONS
# ============================================================================
# NumPy operations are element-wise by default: each operation applies to
# corresponding elements. This is very efficient due to vectorization.

print("\n" + "=" * 70)
print("6. ARRAY OPERATIONS")
print("=" * 70)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 6.1 Arithmetic operations (element-wise)
# Each operation is applied to corresponding elements independently
# Much faster than using loops!
print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")        # Element-wise addition: [1+5, 2+6, 3+7, 4+8]
print(f"a - b = {a - b}")        # Element-wise subtraction
print(f"a * b = {a * b}")        # Element-wise multiplication
print(f"a / b = {a / b}")        # Element-wise division
print(f"a ** 2 = {a ** 2}")      # Element-wise power: [1^2, 2^2, 3^2, 4^2]
print(f"a // b = {a // b}")      # Element-wise floor division
print(f"a % b = {a % b}")        # Element-wise modulo

# 6.2 Broadcasting
# Broadcasting allows operations between arrays of different shapes
# NumPy automatically expands smaller arrays to match larger ones
print(f"\na + 10 = {a + 10}")     # Scalar added to each element: [11, 12, 13, 14]
print(f"a * 2 = {a * 2}")        # Scalar multiplied with each element

# 6.3 Comparison operations
# Return boolean arrays showing which elements satisfy the condition
print(f"\na > 2 = {a > 2}")       # [False, False, True, True]
print(f"a == 3 = {a == 3}")      # [False, False, True, False]
print(f"a != b = {a != b}")      # [True, True, True, True]

# 6.4 Logical operations
# Combine multiple boolean conditions
x = np.array([True, False, True, False])
y = np.array([True, True, False, False])
print(f"\nx = {x}")
print(f"y = {y}")
print(f"np.logical_and(x, y) = {np.logical_and(x, y)}")  # AND: both must be True
print(f"np.logical_or(x, y) = {np.logical_or(x, y)}")    # OR: at least one True
print(f"np.logical_not(x) = {np.logical_not(x)}")        # NOT: opposite of each element

# ============================================================================
# 7. AGGREGATION FUNCTIONS
# ============================================================================
# Aggregation functions combine multiple elements into a single value.
# Can be applied to the entire array or along specific axes.

print("\n" + "=" * 70)
print("7. AGGREGATION FUNCTIONS")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"\nArray: {arr}")
print(f"sum: {np.sum(arr)}")              # Total: 1+2+...+10 = 55
print(f"mean: {np.mean(arr)}")            # Average: 55/10 = 5.5
print(f"median: {np.median(arr)}")        # Middle value: 5.5
print(f"std: {np.std(arr)}")              # Standard deviation: measure of spread
print(f"var: {np.var(arr)}")              # Variance: std squared
print(f"min: {np.min(arr)}")              # Minimum value: 1
print(f"max: {np.max(arr)}")              # Maximum value: 10
print(f"argmin: {np.argmin(arr)}")        # Index of minimum: 0
print(f"argmax: {np.argmax(arr)}")        # Index of maximum: 9

# 7.2 Along axis
# axis=0: reduce rows (vertically)
# axis=1: reduce columns (horizontally)
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D Array:\n{arr2d}")
print(f"sum(axis=0): {np.sum(arr2d, axis=0)}")    # Sum each column: [1+4+7, 2+5+8, 3+6+9]
print(f"sum(axis=1): {np.sum(arr2d, axis=1)}")    # Sum each row: [1+2+3, 4+5+6, 7+8+9]
print(f"mean(axis=0): {np.mean(arr2d, axis=0)}")  # Average of each column
print(f"mean(axis=1): {np.mean(arr2d, axis=1)}")  # Average of each row

# ============================================================================
# 8. MATHEMATICAL FUNCTIONS
# ============================================================================
# NumPy provides universal functions (ufuncs) for mathematical operations.
# These are applied element-wise to entire arrays.

print("\n" + "=" * 70)
print("8. MATHEMATICAL FUNCTIONS")
print("=" * 70)

arr = np.array([0, 1, 2, 3, 4])
print(f"\nArray: {arr}")
print(f"sqrt: {np.sqrt(arr)}")           # Square root: [0, 1, 1.41..., 1.73..., 2]
print(f"exp: {np.exp(arr)}")             # Exponential e^x: [1, 2.71..., 7.38..., 20.08..., 54.59...]
print(f"log: {np.log(arr + 1)}")         # Natural logarithm (log of arr+1 to avoid log(0))
print(f"sin: {np.sin(arr)}")             # Sine (in radians)
print(f"cos: {np.cos(arr)}")             # Cosine (in radians)
print(f"tan: {np.tan(arr)}")             # Tangent (in radians)
print(f"abs: {np.abs(np.array([-1, -2, -3]))}")  # Absolute value: [1, 2, 3]

# ============================================================================
# 9. CONCATENATION AND STACKING
# ============================================================================
# Concatenation combines arrays along existing dimensions.
# Stacking combines arrays by creating a new dimension.

print("\n" + "=" * 70)
print("9. CONCATENATION AND STACKING")
print("=" * 70)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 9.1 Concatenate
# Joins arrays end-to-end along an existing axis
print(f"\na = {a}")
print(f"b = {b}")
print(f"np.concatenate([a, b]) = {np.concatenate([a, b])}")  # [1, 2, 3, 4, 5, 6]

# 9.2 Stack
# Creates a new dimension by stacking arrays
print(f"np.stack([a, b]) =\n{np.stack([a, b])}")            # 2×3 array with a as row 1, b as row 2
print(f"np.vstack([a, b]) =\n{np.vstack([a, b])}")          # Vertical stack (row-wise): same as stack
print(f"np.hstack([a, b]) = {np.hstack([a, b])}")           # Horizontal stack: same as concatenate for 1D

# 9.3 2D concatenation
# Specify axis: 0 (rows) or 1 (columns)
a2d = np.array([[1, 2], [3, 4]])
b2d = np.array([[5, 6], [7, 8]])
print(f"\na2d =\n{a2d}")
print(f"b2d =\n{b2d}")
print(f"concatenate along axis=0:\n{np.concatenate([a2d, b2d], axis=0)}")  # Stack b2d below a2d
print(f"concatenate along axis=1:\n{np.concatenate([a2d, b2d], axis=1)}")  # Place b2d to right of a2d

# ============================================================================
# 10. SPLITTING ARRAYS
# ============================================================================
# Splitting divides an array into multiple subarrays.
# This is the opposite of concatenation.

print("\n" + "=" * 70)
print("10. SPLITTING ARRAYS")
print("=" * 70)

arr = np.arange(12)  # [0, 1, 2, ..., 11]
print(f"\nArray: {arr}")

# 10.1 split
# Divides array into N equal parts
split_result = np.split(arr, 3)  # Divide into 3 equal parts
print(f"np.split(arr, 3):")
for i, sub_arr in enumerate(split_result):
    print(f"  Part {i}: {sub_arr}")  # Each part has 4 elements

# 10.2 2D split
# Can split along different axes
arr2d = np.arange(12).reshape(3, 4)  # 3×4 array
print(f"\n2D Array:\n{arr2d}")
print(f"np.split along axis=0:")
for i, sub_arr in enumerate(np.split(arr2d, 3, axis=0)):  # Split into 3 parts along rows
    print(f"  Part {i}:\n{sub_arr}")  # Each part is 1×4

# ============================================================================
# 11. SORTING AND SEARCHING
# ============================================================================
# Sorting arranges elements in order.
# Searching functions find elements or positions matching criteria.

print("\n" + "=" * 70)
print("11. SORTING AND SEARCHING")
print("=" * 70)

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"\nArray: {arr}")
print(f"sort: {np.sort(arr)}")                                    # Returns sorted array: [1, 1, 2, 3, 4, 5, 6, 9]
print(f"argsort: {np.argsort(arr)}")                              # Returns indices that would sort the array
print(f"searchsorted([1, 2, 3, 4, 5], 3): {np.searchsorted(np.array([1, 2, 3, 4, 5]), 3)}")  # Position where 3 fits

# ============================================================================
# 12. COPYING ARRAYS
# ============================================================================
# Understanding references, views, and copies is crucial for avoiding bugs.
# Modifying one array may or may not affect another depending on the type of copy.

print("\n" + "=" * 70)
print("12. COPYING ARRAYS")
print("=" * 70)

# 12.1 Assignment (reference)
# Assignment does NOT create a copy - just creates another reference to same data
# Changes to arr2 directly change arr
arr = np.array([1, 2, 3])
arr2 = arr  # arr2 now points to the same array as arr
print(f"\nOriginal: {arr}")
arr2[0] = 99  # Modifying arr2 also modifies arr!
print(f"After arr2[0] = 99: {arr}")  # arr has changed too!
print(f"arr is arr2: {arr is arr2}")  # True - they're the same object

# 12.2 View (shallow copy)
# A view shares the same underlying data but is a different array object
# Changes to view affect original and vice versa
arr = np.array([1, 2, 3])
arr_view = arr.view()  # Creates a view
print(f"\nOriginal: {arr}")
arr_view[0] = 99  # Modify the view
print(f"After arr_view[0] = 99: {arr}")  # Original changed too (same data)
print(f"arr is arr_view: {arr is arr_view}")  # False - different objects

# 12.3 Copy (deep copy)
# copy() creates a completely independent array with its own data
# Changes to copy do NOT affect original
arr = np.array([1, 2, 3])
arr_copy = arr.copy()  # Creates independent copy
print(f"\nOriginal: {arr}")
arr_copy[0] = 99  # Modify the copy
print(f"After arr_copy[0] = 99: {arr}")  # Original unchanged!
print(f"arr is arr_copy: {arr is arr_copy}")  # False - different objects with different data

# ============================================================================
# 13. USEFUL TECHNIQUES
# ============================================================================
# Advanced techniques for common operations in numerical computing.

print("\n" + "=" * 70)
print("13. USEFUL TECHNIQUES")
print("=" * 70)

# 13.1 Unique elements
# Find and count unique values in array
arr = np.array([1, 2, 2, 3, 3, 3, 4])
print(f"\nArray: {arr}")
print(f"np.unique(arr) = {np.unique(arr)}")  # Returns [1, 2, 3, 4] - each element once

# 13.2 Matrix multiplication
# Different from element-wise multiplication (*)
# Use dot product or @ operator for matrix operations
a = np.array([[1, 2], [3, 4]])  # 2×2 matrix
b = np.array([[5, 6], [7, 8]])  # 2×2 matrix
print(f"\na =\n{a}")
print(f"b =\n{b}")
print(f"np.dot(a, b) =\n{np.dot(a, b)}")  # Matrix multiplication (dot product)
print(f"a @ b =\n{a @ b}")                 # @ is shorthand for matrix multiplication

# 13.3 Element-wise vs aggregation
# Important distinction in NumPy operations
arr = np.array([1, 2, 3])
print(f"\nArray: {arr}")
print(f"arr.sum() = {arr.sum()}")    # Aggregation: returns single value (6)
print(f"arr + arr = {arr + arr}")    # Element-wise: returns array [2, 4, 6]

print("\n" + "=" * 70)
print("Tutorial Complete!")
print("=" * 70)
