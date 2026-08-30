import conftest_paths  # noqa: F401
import unittest
import osplat


class TestTotalRam(unittest.TestCase):
    def test_returns_positive_or_zero(self):
        n = osplat.total_ram_bytes()
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)

    def test_parse_meminfo(self):
        text = "MemTotal:       32791234 kB\nMemFree: 100 kB\n"
        self.assertEqual(osplat.parse_meminfo(text), 32791234 * 1024)

    def test_parse_meminfo_missing(self):
        self.assertEqual(osplat.parse_meminfo("nope\n"), 0)


class TestAvailableRam(unittest.TestCase):
    """MemAvailable, the number a load decision needs.

    MemTotal says how big the box is; only MemAvailable says how much a newly
    loaded model could still be given - and llama.cpp's --fit reads device
    memory while assuming system memory is unlimited, so on a unified-memory
    APU this is the only host-side brake there is.
    """

    MEMINFO = ("MemTotal:       32791234 kB\n"
               "MemFree:          100000 kB\n"
               "MemAvailable:    8234567 kB\n"
               "SwapFree:         10 kB\n")

    def test_parses_memavailable_in_bytes(self):
        self.assertEqual(osplat.parse_mem_available(self.MEMINFO), 8234567 * 1024)

    def test_does_not_mistake_memfree_or_swapfree_for_memavailable(self):
        self.assertEqual(osplat.parse_mem_available("MemFree: 1 kB\nSwapFree: 2 kB\n"), 0)

    def test_missing_is_zero_not_a_crash(self):
        self.assertEqual(osplat.parse_mem_available("MemTotal: 1 kB\n"), 0)

    def test_available_never_exceeds_total_on_a_real_box(self):
        total, avail = osplat.total_ram_bytes(), osplat.available_ram_bytes()
        self.assertGreaterEqual(avail, 0)
        if total and avail:
            self.assertLessEqual(avail, total)


if __name__ == "__main__":
    unittest.main()
