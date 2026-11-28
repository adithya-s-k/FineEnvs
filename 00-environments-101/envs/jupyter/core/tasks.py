"""
Task definitions for Jupyter Agent ORS training.

Same 46 tasks as openenv/dataset.py. Kept in a separate file so
dataset.py can import without pulling in ORS server dependencies.
"""

import re


def matches_expected(expected: str, output: str) -> bool:
    """Match expected_output on token boundaries to avoid substring false positives."""
    expected = (expected or "").strip()
    output = (output or "").strip()
    if not expected or not output:
        return False
    return bool(re.search(rf"(?<![\w.]){re.escape(expected)}(?!\w|\.\d)", output))

TASKS = [
    {"task": "Install the 'sympy' library using pip, then use sympy.isprime to count how many numbers between 900 and 1000 are prime. Print the count.", "expected_output": "14"},
    {"task": "Install 'scipy' using pip. Generate 500 random numbers from a normal distribution (mean=100, std=15) using numpy seed 0. Use scipy.stats.describe to get the variance. Print the variance rounded to 1 decimal place.", "expected_output": "224.6"},
    {"task": "Step 1: Create a CSV file 'employees.csv' with columns name,department,salary and these rows: Alice,Engineering,95000 / Bob,Marketing,72000 / Carol,Engineering,88000 / Dave,Marketing,68000 / Eve,Engineering,102000. Step 2: Read it with pandas. Step 3: Print the average salary in the Engineering department rounded to the nearest integer.", "expected_output": "95000"},
    {"task": 'Write a JSON file called \'config.json\' containing {"model": "qwen3", "layers": 36, "hidden_size": 2560}. Then read it back, multiply the hidden_size by the number of layers, and print the result.', "expected_output": "92160"},
    {"task": "Write a function to find all twin prime pairs (primes that differ by 2) less than 100. Print the total number of twin prime pairs found.", "expected_output": "8"},
    {"task": "Implement the Sieve of Eratosthenes to find all primes up to 200. Then find the largest prime gap (difference between consecutive primes) in that range. Print the largest gap.", "expected_output": "14"},
    {"task": "Create a 10x10 multiplication table as a numpy array where element [i,j] = (i+1)*(j+1). Print the sum of all elements in the table.", "expected_output": "3025"},
    {"task": "Using numpy with seed 42: generate 100 random integers between 1-100, compute the 25th percentile, 50th percentile (median), and 75th percentile. Print the IQR (75th - 25th percentile) rounded to 1 decimal place.", "expected_output": "53.0"},
    {"task": "Create a pandas DataFrame with 50 rows and columns 'x' (integers 1-50) and 'y' (x squared plus a random noise from numpy seed 7, noise = numpy.random.normal(0, 50, 50)). Compute the Pearson correlation between x and y. Print the correlation rounded to 3 decimal places.", "expected_output": "0.965"},
    {"task": "Use a shell command to check the Python version, then write a Python script that parses the version string and prints just the major version number (e.g., if Python 3.13.1, print 3).", "expected_output": "3"},
    {"task": "Try to compute 1/0 in Python. When you get a ZeroDivisionError, catch it and instead compute the factorial of 12. Print the factorial result.", "expected_output": "479001600"},
    {"task": "Write a recursive function to compute the nth Fibonacci number. First test it with n=10 (should print 55). Then test with n=30 and print the result.", "expected_output": "832040"},
    {"task": "Calculate the sum of all prime numbers less than 50 and print the result.", "expected_output": "328"},
    {"task": "Calculate 2^20 and print the result.", "expected_output": "1048576"},
    {"task": "Calculate the factorial of 10 and print the result.", "expected_output": "3628800"},
    {"task": "Calculate the sum of integers from 1 to 100 and print the result.", "expected_output": "5050"},
    {"task": "Calculate the greatest common divisor of 48 and 180 and print the result.", "expected_output": "12"},
    {"task": "Calculate how many integers between 1 and 100 are divisible by 3 or 5 and print the count.", "expected_output": "47"},
    {"task": "Calculate the 20th Fibonacci number (starting from F(1)=1, F(2)=1) and print it.", "expected_output": "6765"},
    {"task": "Calculate the sum of digits of 99999 and print the result.", "expected_output": "45"},
    {"task": "Count the number of vowels (a, e, i, o, u) in the string 'reinforcement learning' and print the count.", "expected_output": "7"},
    {"task": "Reverse the string 'hello world' and print the result.", "expected_output": "dlrow olleh"},
    {"task": "Count the number of unique characters in 'mississippi' and print the count.", "expected_output": "4"},
    {"task": "Convert 'hello world' to uppercase and print it.", "expected_output": "HELLO WORLD"},
    {"task": "Count the number of words in 'the quick brown fox jumps over the lazy dog' and print the count.", "expected_output": "9"},
    {"task": "Find the longest word in 'the quick brown fox jumps over the lazy dog' and print it.", "expected_output": "quick"},
    {"task": "Create a numpy array of integers from 1 to 10 inclusive, compute the mean, and print it.", "expected_output": "5.5"},
    {"task": "Create a numpy array [1, 4, 9, 16, 25], compute the standard deviation, and print the result rounded to 2 decimal places.", "expected_output": "8.0"},
    {"task": "Generate the first 10 square numbers (1, 4, 9, ..., 100) using numpy and print their sum.", "expected_output": "385"},
    {"task": "Create a 3x3 identity matrix using numpy and print the sum of all its elements.", "expected_output": "3.0"},
    {"task": "Calculate the dot product of vectors [1, 2, 3] and [4, 5, 6] using numpy and print the result.", "expected_output": "32"},
    {"task": "Create a pandas DataFrame with columns 'name' = ['Alice', 'Bob', 'Carol'] and 'score' = [95, 82, 91]. Print the mean of the 'score' column rounded to 1 decimal place.", "expected_output": "89.3"},
    {"task": "Create a pandas DataFrame with column 'values' = [10, 20, 30, 40, 50]. Print the sum of the column.", "expected_output": "150"},
    {"task": "Create a pandas DataFrame with columns A=[1,2,3,4,5] and B=[5,4,3,2,1]. Print the correlation between A and B rounded to 1 decimal place.", "expected_output": "-1.0"},
    {"task": "Create a pandas Series with values [3, 1, 4, 1, 5, 9, 2, 6]. Print the median.", "expected_output": "3.5"},
    {"task": "Create a dictionary with keys 'a'=1, 'b'=2, 'c'=3. Print the sum of all values.", "expected_output": "6"},
    {"task": "Create a list of the first 10 even numbers (2, 4, ..., 20) and print their sum.", "expected_output": "110"},
    {"task": "Create a set from the list [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] and print the length of the set.", "expected_output": "4"},
    {"task": "Sort the list [5, 3, 8, 1, 9, 2, 7, 4, 6] in ascending order and print it.", "expected_output": "[1, 2, 3, 4, 5, 6, 7, 8, 9]"},
    {"task": "Check if the number 997 is prime and print True or False.", "expected_output": "True"},
    {"task": "Count the number of prime numbers between 1 and 100 and print the count.", "expected_output": "25"},
    {"task": "Find the second largest number in the list [15, 3, 42, 8, 27, 42, 1] and print it.", "expected_output": "27"},
    {"task": "Convert the decimal number 255 to binary (as a string without '0b' prefix) and print it.", "expected_output": "11111111"},
    {"task": "Write the text 'hello from jupyter' to a file called 'test.txt', then read it back and print the contents.", "expected_output": "hello from jupyter"},
    {"task": "Create a list of numbers from 1 to 20, filter out the odd numbers, square the remaining even numbers, and print the sum.", "expected_output": "1540"},
    {"task": "Create a pandas DataFrame with 3 columns: 'x' = range(1,6), 'y' = [2,4,6,8,10], 'z' = [1,3,5,7,9]. Print the column with the highest sum.", "expected_output": "y"},
]
