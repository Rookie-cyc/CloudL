def sort_list(lst):
    return sorted(lst)

def testMLOpsSortList():
    # Test Case 1: Normal list
    assert sort_list([3, 1, 2]) == [1, 2, 3]
    # Test Case 2: List with negative numbers
    assert sort_list([-1, -3, 2, 0]) == [-3, -1, 0, 2]
    # Test Case 3: Empty list
    assert sort_list([]) == []
    # Test Case 4: List with repeated elements
    assert sort_list([4, 2, 2, 3, 4]) == [2, 2, 3, 4, 4]
    # Test Case 5: Already sorted list
    assert sort_list([1, 2, 3, 4]) == [1, 2, 3, 4]
    print("All MLOps sort_list tests passed successfully!")

testMLOpsSortList()
