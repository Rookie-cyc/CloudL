def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def testMLOpsIsPrime():
    # Test Case 1: Prime number
    assert is_prime(7) == True
    # Test Case 2: Non-prime number
    assert is_prime(8) == False
    # Test Case 3: Negative number
    assert is_prime(-5) == False
    # Test Case 4: Zero
    assert is_prime(0) == False
    # Test Case 5: One
    assert is_prime(1) == False
    # Test Case 6: Large prime
    assert is_prime(97) == True
    print("All MLOps is_prime tests passed successfully!")

testMLOpsIsPrime()