from saev.helpers import expand


def test_expand():
    cfg = {"lr": [1, 2, 3]}
    expected = [{"lr": 1}, {"lr": 2}, {"lr": 3}]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_two_fields():
    cfg = {"lr": [1, 2], "wd": [3, 4]}
    expected = [
        {"lr": 1, "wd": 3},
        {"lr": 1, "wd": 4},
        {"lr": 2, "wd": 3},
        {"lr": 2, "wd": 4},
    ]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_nested():
    cfg = {"sae": {"dim": [1, 2, 3]}}
    expected = [{"sae": {"dim": 1}}, {"sae": {"dim": 2}}, {"sae": {"dim": 3}}]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_nested_and_unnested():
    cfg = {"sae": {"dim": [1, 2]}, "lr": [3, 4]}
    expected = [
        {"sae": {"dim": 1}, "lr": 3},
        {"sae": {"dim": 1}, "lr": 4},
        {"sae": {"dim": 2}, "lr": 3},
        {"sae": {"dim": 2}, "lr": 4},
    ]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_nested_and_unnested_backwards():
    cfg = {"a": [False, True], "b": {"c": [False, True]}}
    expected = [
        {"a": False, "b": {"c": False}},
        {"a": False, "b": {"c": True}},
        {"a": True, "b": {"c": False}},
        {"a": True, "b": {"c": True}},
    ]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_multiple():
    cfg = {"a": [1, 2, 3], "b": {"c": [4, 5, 6]}}
    expected = [
        {"a": 1, "b": {"c": 4}},
        {"a": 1, "b": {"c": 5}},
        {"a": 1, "b": {"c": 6}},
        {"a": 2, "b": {"c": 4}},
        {"a": 2, "b": {"c": 5}},
        {"a": 2, "b": {"c": 6}},
        {"a": 3, "b": {"c": 4}},
        {"a": 3, "b": {"c": 5}},
        {"a": 3, "b": {"c": 6}},
    ]
    actual = list(expand(cfg))

    assert expected == actual
