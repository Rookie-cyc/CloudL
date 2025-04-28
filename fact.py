def factorial(n):
if n < 0:
return None # Factorial is not defined for negative numbers
if n == 0 or n == 1:
return 1
result = 1
for i in range(2, n + 1):
result *= i
return result
def testMLOpsFactorial():
# Test Case 1: Factorial of a positive number
trainingData1 = 5
expectedOutput1 = 120
assert factorial(trainingData1) == expectedOutput1
# Test Case 2: Factorial of zero
trainingData2 = 0
expectedOutput2 = 1
assert factorial(trainingData2) == expectedOutput2
# Test Case 3: Factorial of one
trainingData3 = 1
expectedOutput3 = 1
assert factorial(trainingData3) == expectedOutput3
# Test Case 4: Factorial of a larger number
trainingData4 = 7
expectedOutput4 = 5040
assert factorial(trainingData4) == expectedOutput4
# Test Case 5: Factorial of a negative number
trainingData5 = -3
expectedOutput5 = None # Expecting None as factorial is undefined for negatives
assert factorial(trainingData5) == expectedOutput5
print("All MLOps factorial tests passed successfully!")
testMLOpsFactorial()