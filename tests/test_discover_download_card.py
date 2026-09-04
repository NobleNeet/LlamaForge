import conftest_paths  # noqa: F401
import os, re, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISCOVER_JS = os.path.join(ROOT, "web", "js", "discover.js")


class TestDownloadCardPlacement(unittest.TestCase):
    """The Download card carries the live progress bar, so it belongs directly
    under the "Discover models on huggingface.co" card. A search returns dozens
    of repo rows, and a card placed after #hub-results is only reachable by
    scrolling past every one of them - which is what it used to be."""

    @classmethod
    def setUpClass(cls):
        with open(DISCOVER_JS, encoding="utf-8") as f:
            cls.js = f.read()
        head = cls.js.index("export function loadDiscover()")
        tail = cls.js.index('$("#dl-cancel")', head)
        cls.view = cls.js[head:tail]      # just the view-discover template

    def test_card_sits_between_the_search_card_and_the_results(self):
        markers = ['<h3>Discover models on huggingface.co</h3>',
                   'class="card" id="hub-dlcard"',
                   'id="hub-results"']
        for marker in markers:
            self.assertIn(marker, self.view, marker)
        order = [self.view.index(m) for m in markers]
        self.assertEqual(sorted(order), order,
                         "Download card must follow the search card and precede the results")

    def test_card_starts_hidden(self):
        self.assertIn('<div class="card" id="hub-dlcard" style="display:none">', self.view)

    def test_search_results_still_render_last(self):
        """Moving the card must not move where results are painted into."""
        self.assertIn('setHTML($("#hub-results"), `<div class="list">', self.js)

    def test_every_download_entry_point_reveals_the_card(self):
        """GGUF start, the Resume re-poll, and the vLLM/WSL transfer share the
        one card, so all three have to unhide it where it now sits."""
        self.assertEqual(3, len(re.findall(r'\$\("#hub-dlcard"\)\.style\.display = ""', self.js)))


if __name__ == "__main__":
    unittest.main()
