def sum_nested_list(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += sum_nested_list(item)
        else:
            total += item
    return total

def testMLOpsSumNestedList():
    # Test Case 1: Flat list
    assert sum_nested_list([1, 2, 3]) == 6
    # Test Case 2: Nested list
    assert sum_nested_list([1, [2, 3], 4]) == 10
    # Test Case 3: Deep nesting
    assert sum_nested_list([1, [2, [3, [4]]]]) == 10
    # Test Case 4: List with negative numbers
    assert sum_nested_list([-1, [2, -3], 4]) == 2
    # Test Case 5: Empty list
    assert sum_nested_list([]) == 0
    print("All MLOps sum_nested_list tests passed successfully!")

testMLOpsSumNestedList()
