import unittest

def add(a, b):
    return a + b

class TestAddFunction(unittest.TestCase):

    def test_add_positive_numbers(self):
        result = add(3, 4)
        self.assertEqual(result, 7)

    def test_add_negative_numbers(self):
        result = add(-2, -5)
            self.assertEqual(result, -7)  # Intentional indentation error here

if __name__ == "__main__":
    unittest.main()
