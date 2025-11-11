import pytest
from hypothesis import given
from hypothesis import strategies as st

from saev import viz


def test_parse_color_hex_lowercase():
    color = viz.parse_color("#ff8800")
    assert color == pytest.approx((1.0, 136 / 255, 0.0))


def test_parse_color_hex_uppercase():
    color = viz.parse_color("#ABCDEF")
    assert color == pytest.approx((171 / 255, 205 / 255, 239 / 255))


def test_parse_color_hex_with_whitespace():
    color = viz.parse_color("  #010203  ")
    assert color == pytest.approx((1 / 255, 2 / 255, 3 / 255))


def test_parse_color_rgb_float_values():
    color = viz.parse_color("rgb(0.1, 0.2, 0.3)")
    assert color == pytest.approx((0.1, 0.2, 0.3))


def test_parse_color_rgb_int_values():
    color = viz.parse_color("rgb(255, 128, 0)")
    assert color == pytest.approx((1.0, 128 / 255, 0.0))


def test_parse_color_rgb_keyword_case_insensitive():
    color = viz.parse_color("RGB(0.4,0.5,0.6)")
    assert color == pytest.approx((0.4, 0.5, 0.6))


def test_parse_color_rejects_short_hex():
    with pytest.raises(AssertionError):
        viz.parse_color("#123")


def test_parse_color_rejects_non_rgb_input():
    with pytest.raises(AssertionError):
        viz.parse_color("hsl(0, 1, 0.5)")


def test_parse_color_rejects_component_overflow():
    with pytest.raises(AssertionError):
        viz.parse_color("rgb(300, 0, 0)")


def test_parse_color_rejects_negative_component():
    with pytest.raises(AssertionError):
        viz.parse_color("rgb(-1, 0, 0)")


@given(
    st.tuples(
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    )
)
def test_parse_color_hex_property(rgb_triplet):
    r, g, b = rgb_triplet
    line = f"#{r:02x}{g:02x}{b:02x}"
    color = viz.parse_color(line)
    expected = (r / 255, g / 255, b / 255)
    assert color == pytest.approx(expected)


@given(
    st.tuples(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
)
def test_parse_color_rgb_float_property(rgb_triplet):
    r, g, b = rgb_triplet
    line = f"rgb({r:.6f}, {g:.6f}, {b:.6f})"
    color = viz.parse_color(line)
    channels = tuple(
        float(part.strip())
        for part in line[line.index("(") + 1 : line.rindex(")")].split(",")
    )
    assert color == pytest.approx(channels)


@given(
    st.tuples(
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    )
)
def test_parse_color_rgb_int_property(rgb_triplet):
    r, g, b = rgb_triplet
    line = f"rgb({r}, {g}, {b})"
    color = viz.parse_color(line)
    max_chan = max(rgb_triplet)
    if max_chan <= 1:
        expected = tuple(float(chan) for chan in rgb_triplet)
    else:
        expected = (r / 255, g / 255, b / 255)
    assert color == pytest.approx(expected)
