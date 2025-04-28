def is_palindrome(text):
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum())
    return cleaned_text == cleaned_text[::-1]

def testMLOpsPalindrome():
    # Test Case 1: Even length palindrome
    assert is_palindrome("noon") == True
    # Test Case 2: Odd length palindrome
    assert is_palindrome("level") == True
    # Test Case 3: Palindrome with mixed characters
    assert is_palindrome("A man, a plan, a canal: Panama") == True
    # Test Case 4: Non-palindrome
    assert is_palindrome("hello") == False
    # Test Case 5: Palindrome with numbers
    assert is_palindrome("12321") == True
    # Test Case 6: Empty string
    assert is_palindrome("") == True
    # Test Case 7: Special characters only
    assert is_palindrome("!@##@!") == True

    print("All MLOps palindrome tests passed successfully!")

# Run the test
testMLOpsPalindrome()
