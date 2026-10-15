"""Scientific protocol regression tests: leakage, censoring, grouping and plan identity."""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp" / "exp3_e2e"))
from protocol import assign_folds, cluster_interval, plan_fingerprint, validate_complete_pack
from models.dace.plan_utils import fit_node_statistics, NodeFeaturizer


def plan(cost, kind="Seq Scan"):
    return {"Plan": {"Node Type": kind, "Total Cost": cost, "Plan Rows": cost,
                     "Actual Total Time": 1000000 - cost}}


def test_training_statistics_do_not_read_labels_or_test_nodes():
    training = [plan(1), plan(3), plan(5)]
    stats = fit_node_statistics(training)
    test = plan(10**9, "Unseen Operator")
    assert stats["Total Cost"]["center"] == 3
    assert "Unseen Operator" not in stats["node_types"]["value_dict"]
    assert NodeFeaturizer(stats).node_vector(test["Plan"])[0] == 0
    for p in training:
        p["Plan"]["Actual Total Time"] = -123456
    assert fit_node_statistics(training) == stats


def test_plan_fingerprint_ignores_observations_but_keeps_predicates_and_order():
    p = plan(5)
    p["Plan"]["Plans"] = [plan(1)["Plan"], plan(3)["Plan"]]
    other = copy.deepcopy(p)
    other["Plan"]["Actual Total Time"] = 99
    assert plan_fingerprint(p) == plan_fingerprint(other)
    other["Plan"]["Plans"].reverse()
    assert plan_fingerprint(p) != plan_fingerprint(other)
    other = copy.deepcopy(p)
    other["Plan"]["Filter"] = "a = 1"
    assert plan_fingerprint(p) != plan_fingerprint(other)


def test_timeout_remains_in_complete_candidate_set():
    from common import candidates
    p = {"query_id": "x", "expected_hint_ids": [0, 1], "results": [
        {"hint_id": 0, "plan": plan(1), "actual_exec_time_ms": 4},
        {"hint_id": 1, "plan": plan(2), "actual_exec_time_ms": None,
         "evaluation_ms": 32000, "censored": True}]}
    validate_complete_pack(p)
    assert [r[2] for r in candidates(p)] == [4, 32000]
    del p["results"][1]["plan"]
    with pytest.raises(ValueError, match="no planning-time plan"):
        validate_complete_pack(p)


def test_template_fold_membership_is_independent_of_training_seed():
    packs = [{"query_id": f"{i}{suffix}_job"} for i in range(1, 11) for suffix in ("a", "b")]
    membership = assign_folds(packs, folds=5, split_seed=42)
    assert all(membership[i] == membership[i + 1] for i in range(0, 20, 2))
    assert set(membership.values()) == set(range(5))
    assert membership == assign_folds(packs, folds=5, split_seed=42)


def test_bootstrap_resamples_templates_and_preserves_query_weighting():
    values = np.log([2, 2, 2, 0.5])
    result = cluster_interval(values, ["a", "a", "a", "b"], samples=5000, seed=9)
    assert result["geomean"] == pytest.approx(np.sqrt(2))
    assert result["template_equal_weight_geomean"] == pytest.approx(1)
    assert result["ci95_low"] == pytest.approx(0.5)
    assert result["ci95_high"] == pytest.approx(2)


def test_tree_convolution_receives_original_parent_child_edges():
    from run_kfold import to_graph
    p = plan(2)
    p["Plan"]["Plans"] = [plan(1)["Plan"]]
    fz = NodeFeaturizer(fit_node_statistics([p]))
    graph = to_graph(p, fz, 0.1, 0, 0, 1, parent_to_child=True)
    assert graph.edge_index.tolist() == [[0], [1]]


def test_collector_retains_every_timeout_arm(tmp_path, monkeypatch):
    import json
    import psycopg2
    import collect_benchmark as collector

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, *args): pass

    class Connection:
        def cursor(self): return Cursor()

    def fake_explain(cur, sql, execute=False):
        if execute:
            raise psycopg2.errors.QueryCanceled("test timeout")
        return plan(1)

    monkeypatch.setattr(collector, "explain", fake_explain)
    collector.collect_query(Connection(), {"query_id": "timeout_test", "sql": "SELECT 1", "template": "x"},
                            tmp_path, {"timeout_ms": 10, "seed": 42, "warmup": 1, "repeats": 3}, "testhash")
    pack = json.loads((tmp_path / "q_timeout_test.json").read_text())
    validate_complete_pack(pack)
    assert len(pack["results"]) == 49
    assert pack["latency_metric"] == "capped_statement_wall_ms"
    assert all(r["censored"] and r["evaluation_ms"] == 10 and r["actual_exec_time_ms"] is None
               and len(r["measurements"]) == 3 for r in pack["results"])


def test_controlled_tree_convolution_rejects_silent_child_loss():
    from run_kfold import to_graph
    p = plan(4)
    p["Plan"]["Plans"] = [plan(i)["Plan"] for i in (1, 2, 3)]
    fz = NodeFeaturizer(fit_node_statistics([p]))
    with pytest.raises(ValueError, match="requires binary plans"):
        to_graph(p, fz, 0, 0, 0, 1, parent_to_child=True)


@pytest.mark.parametrize("arity", [0, 1, 2, 3, 4, 8])
def test_ordered_binary_conversion_is_reversible_and_nonmutating(arity):
    from adapters.plan_representation import represent_plan, restore_plan, GROUP_MARKER
    from run_kfold import walk
    p = plan(100)
    p["Plan"]["Plans"] = [dict(plan(i + 1)["Plan"], **{"Subplan Name": f"SubPlan {i}"})
                           for i in range(arity)]
    if arity:
        p["Plan"]["Plans"][-1]["Plans"] = [plan(i)["Plan"] for i in (7, 8, 9)]
    before = copy.deepcopy(p)
    binary = represent_plan(p, "ordered-binary-v1")
    assert p == before
    assert restore_plan(binary) == p
    assert all(len(n.get("Plans", [])) <= 2 for n in walk(binary["Plan"]))
    assert sum(not n.get(GROUP_MARKER, False) for n in walk(binary["Plan"])) == len(list(walk(p["Plan"])))


def test_binary_representation_is_identity_for_binary_inputs_and_preserves_edges():
    from adapters.plan_representation import represent_plan
    from run_kfold import to_graph
    import torch
    p = plan(4)
    p["Plan"]["Plans"] = [plan(1)["Plan"], plan(2)["Plan"]]
    binary = represent_plan(p, "ordered-binary-v1")
    assert binary == p and binary is not p
    fz = NodeFeaturizer(fit_node_statistics([p]))
    for tree in (False, True):
        a = to_graph(p, fz, 0, 0, 0, 1, parent_to_child=tree)
        b = to_graph(binary, fz, 0, 0, 0, 1, parent_to_child=tree)
        assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index)


def test_later_subplan_reaches_tree_convolution_root():
    from adapters.plan_representation import represent_plan
    from run_kfold import to_graph
    from models.gnto.graph_encoder import TreeConv
    import torch
    p = plan(10)
    p["Plan"]["Plans"] = [plan(i)["Plan"] for i in (1, 2, 3, 4)]
    binary = represent_plan(p, "ordered-binary-v1")
    fz = NodeFeaturizer(fit_node_statistics([binary]))
    graph = to_graph(binary, fz, 0, 0, 0, 1, parent_to_child=True)
    x = torch.ones((graph.num_nodes, 1), requires_grad=True)
    conv = TreeConv(1, 1)
    with torch.no_grad():
        conv.lin.weight.fill_(1)
        conv.lin.bias.zero_()
    h = x
    for _ in range(4):
        h = conv(h, graph.edge_index)
    h[0].sum().backward()
    assert torch.all(x.grad > 0)


def test_extended_native_adapter_retains_unary_scan_and_all_subplans():
    from adapters.plan_representation import represent_plan
    from adapters.native_tree_compat import ExtendedTreeBuilder, PADDING_LABEL
    from run_kfold import walk
    p = plan(20, "Aggregate")
    scan = dict(plan(10)["Plan"], **{"Relation Name": "outer"})
    scan["Plans"] = [dict(plan(5)["Plan"], **{"Relation Name": "sub", "Parent Relationship": "SubPlan"})]
    p["Plan"]["Plans"] = [scan, plan(4)["Plan"], plan(3)["Plan"]]
    binary = represent_plan(p, "ordered-binary-v1")
    stats = lambda n: [n.get("Total Cost", 0), n.get("Plan Rows", 0)]
    builder = ExtendedTreeBuilder(stats, [])
    tree = builder.plan_to_feature_tree(binary["Plan"])
    def real_nodes(node):
        if len(node) == 2:
            return [] if node[1] == PADDING_LABEL else [node]
        return [node] + real_nodes(node[1]) + real_nodes(node[2])
    nodes = real_nodes(tree)
    assert len(nodes) == len(list(walk(binary["Plan"])))
    assert any(n[0][3] == 1 and n[0][7] == 10 for n in nodes)  # outer scan retained
    assert any(n[0][3] == 1 and n[0][7] == 5 for n in nodes)   # its SubPlan retained
    assert tree[0][7] == 20                                  # Aggregate retained
    assert all(np.isfinite(n[0]).all() for n in nodes)
