import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "agx_arm_ctrl"
    / "agx_arm_ctrl_single_node.py"
)


def load_finite_float_list():
    source = SOURCE_PATH.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "finite_float_list"
    )
    namespace = {}
    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name="math")]), function],
        type_ignores=[],
    )
    module = ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["finite_float_list"]


def load_function(name):
    source = SOURCE_PATH.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert name in functions, f"missing function {name}"
    namespace = {}
    module = ast.Module(body=[functions[name]], type_ignores=[])
    module = ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


def test_finite_float_list_normalizes_numeric_values():
    finite_float_list = load_finite_float_list()

    assert finite_float_list([1, 2.5, "3.25"]) == [1.0, 2.5, 3.25]


def test_finite_float_list_rejects_non_finite_or_invalid_values():
    finite_float_list = load_finite_float_list()

    assert finite_float_list([1.0, float("nan")]) is None
    assert finite_float_list([1.0, float("inf")]) is None
    assert finite_float_list([1.0, object()]) is None


def test_normalize_speed_percent_accepts_only_integer_percentages():
    normalize_speed_percent = load_function("normalize_speed_percent")

    assert normalize_speed_percent(6) == 6
    for invalid in (True, 0, 101, 6.0, "6"):
        try:
            normalize_speed_percent(invalid)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid speed percentage: {invalid!r}")


def test_arm_status_publisher_includes_raw_error_code():
    source = SOURCE_PATH.read_text()

    assert "msg.err_status = arm_status.msg.err_code" in source
