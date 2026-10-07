import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_integration")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
