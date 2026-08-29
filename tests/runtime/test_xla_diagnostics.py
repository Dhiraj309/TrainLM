from trainlm.runtime import XlaDiagnostics


class FakeMetrics:

    def counter_names(self):
        return ("aten::nonzero", "aten::custom_op", "OtherCounter")

    def counter_value(self, name):
        return {
            "aten::nonzero": 2,
            "aten::custom_op": 3,
            "OtherCounter": 99,
        }[name]

    def metric_data(self, name):
        return {
            "CompileTime": (4, 0, []),
            "ExecuteTime": (12, 0, []),
        }.get(name, (0, 0, []))

    def short_metrics_report(self):
        return "short report"

    def metrics_report(self):
        return "full report"

    def clear_all(self):
        self.cleared = True


def test_xla_diagnostics_collects_bounded_metrics_and_fallbacks():
    diagnostics = XlaDiagnostics(FakeMetrics(), max_report_chars=128)
    diagnostics.record_hlo("module { entry }")
    diagnostics.record_profile_metadata({"service": "xprof", "step": 4})

    snapshot = diagnostics.snapshot()

    assert snapshot["compile_count"] == 4
    assert snapshot["execute_count"] == 12
    assert snapshot["aten_fallback_count"] == 5
    assert snapshot["aten_fallbacks"] == {
        "aten::custom_op": 3,
        "aten::nonzero": 2,
    }
    assert snapshot["hlo"] == "module { entry }"
    assert snapshot["profile_metadata"]["service"] == "xprof"


def test_xla_diagnostics_truncates_reports_and_can_clear():
    diagnostics = XlaDiagnostics(FakeMetrics(), max_report_chars=128)
    diagnostics.record_hlo("hlo")
    diagnostics.record_profile_metadata(None)

    diagnostics.clear()

    assert getattr(diagnostics.metrics_module, "cleared", False)
    assert diagnostics.snapshot()["profile_metadata"] == {}
