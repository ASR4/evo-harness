from evoharness.core.lineage import LineageGraph


class TestLineageGraph:
    def test_add_single_root(self):
        g = LineageGraph()
        g.add_candidate("000", [])
        assert g.get_parents("000") == []
        assert g.get_roots() == ["000"]

    def test_add_with_parent(self):
        g = LineageGraph()
        g.add_candidate("000", [])
        g.add_candidate("001", ["000"])
        assert g.get_parents("001") == ["000"]
        assert g.get_children("000") == ["001"]

    def test_multi_parent_merge(self):
        g = LineageGraph()
        g.add_candidate("000", [])
        g.add_candidate("001", ["000"])
        g.add_candidate("002", ["000"])
        g.add_candidate("003", ["001", "002"])
        assert sorted(g.get_parents("003")) == ["001", "002"]
        assert "003" in g.get_children("001")
        assert "003" in g.get_children("002")

    def test_get_ancestors(self):
        g = LineageGraph()
        g.add_candidate("000", [])
        g.add_candidate("001", ["000"])
        g.add_candidate("002", ["001"])
        ancestors = g.get_ancestors("002")
        assert "001" in ancestors
        assert "000" in ancestors

    def test_get_ancestors_unknown(self):
        g = LineageGraph()
        assert g.get_ancestors("999") == []

    def test_save_load(self, tmp_path):
        g = LineageGraph()
        g.add_candidate("000", [])
        g.add_candidate("001", ["000"])
        g.add_candidate("002", ["000"])
        path = tmp_path / "lineage.json"
        g.save(path)
        loaded = LineageGraph.load(path)
        assert loaded.get_parents("001") == ["000"]
        assert sorted(loaded.get_children("000")) == ["001", "002"]

    def test_format_lineage(self):
        g = LineageGraph()
        g.add_candidate("000", [])
        g.add_candidate("001", ["000"])
        text = g.format_lineage("001")
        assert "001" in text
        assert "000" in text

    def test_format_unknown(self):
        g = LineageGraph()
        text = g.format_lineage("999")
        assert "unknown" in text
