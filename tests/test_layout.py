import pytest
import kzn_recsys as fease
from kzn_recsys import Format, optimize_layout, GaussianAnomalyDetector

def test_optimize_layout_basic():
    # Tray 0 options
    t0 = {
        "id": 0,
        "options": [
            {"format": "None", "height": 0, "utility": 0.0, "item_count": 0},
            {"format": "Carousel", "height": 2, "utility": 1.5, "item_count": 5},
            {"format": "Banner", "height": 4, "utility": 3.0, "item_count": 1},
        ]
    }

    # Tray 1 options
    t1 = {
        "id": 1,
        "options": [
            {"format": "None", "height": 0, "utility": 0.0, "item_count": 0},
            {"format": "Carousel", "height": 2, "utility": 2.0, "item_count": 5},
            {"format": "Banner", "height": 4, "utility": 4.5, "item_count": 1},
        ]
    }

    trays = [t0, t1]

    # Scenario 1: Max Height = 8.
    # Attempting to select Banner (3.0, 4) + Banner (4.5, 4) is prevented by adjacency.
    # Carousel (2, 1.5) + Banner (4, 4.5) -> total utility = 6.0 (optimal)
    val, formats = optimize_layout(trays, 8)
    assert val == 6.0
    assert formats == [Format.Carousel, Format.Banner]

    # Scenario 2: Max Height = 3.
    # None + Carousel -> total utility = 2.0 (optimal)
    val, formats = optimize_layout(trays, 3)
    assert val == 2.0
    assert formats == [getattr(Format, 'None'), Format.Carousel]

def test_optimize_layout_invalid_inputs():
    with pytest.raises(KeyError):
        # Missing 'id'
        optimize_layout([{"options": []}], 8)

    with pytest.raises(KeyError):
        # Missing 'options'
        optimize_layout([{"id": 0}], 8)

    with pytest.raises(TypeError):
        # Options not a list
        optimize_layout([{"id": 0, "options": "invalid"}], 8)

def test_gaussian_anomaly_detector():
    history = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1002.0, 998.0, 1008.0, 992.0, 1000.0]
    
    # Fit detector from observations
    detector = GaussianAnomalyDetector.fit(history, 3.0)
    assert detector is not None
    assert abs(detector.mean - 1000.0) < 1.0
    assert detector.std < 10.0
    assert detector.low > 950.0
    assert detector.high < 1050.0
    
    # Test check method
    assert detector.check(1000.0, "distinct_users") is True
    assert detector.check(800.0, "distinct_users") is False
    assert detector.check(1200.0, "distinct_users") is False

    # Test constructor round-trip
    manual_detector = GaussianAnomalyDetector(900.0, 1100.0, 1000.0, 50.0, 2.0)
    assert manual_detector.low == 900.0
    assert manual_detector.high == 1100.0
    assert manual_detector.mean == 1000.0
    assert manual_detector.std == 50.0
    assert manual_detector.std_multiplier == 2.0
    assert manual_detector.check(1050.0) is True
    assert manual_detector.check(850.0) is False

def test_optimize_layout_custom_constraints():
    from kzn_recsys import LayoutConstraint
    
    # Tray 0 options
    t0 = {
        "id": 0,
        "options": [
            {"format": "None", "height": 0, "utility": 0.0, "item_count": 0},
            {"format": "Carousel", "height": 2, "utility": 1.5, "item_count": 5},
            {"format": "Banner", "height": 4, "utility": 3.0, "item_count": 1},
        ]
    }

    # Tray 1 options
    t1 = {
        "id": 1,
        "options": [
            {"format": "None", "height": 0, "utility": 0.0, "item_count": 0},
            {"format": "Carousel", "height": 2, "utility": 2.0, "item_count": 5},
            {"format": "Banner", "height": 4, "utility": 4.5, "item_count": 1},
        ]
    }

    trays = [t0, t1]

    # Test 1: DisallowedAtSlot constraint - Disallow Banner in Slot 1
    constraints_1 = [
        LayoutConstraint.disallowed_at_slot(Format.Banner, 1)
    ]
    val, formats = optimize_layout(trays, 8, constraints=constraints_1)
    assert val == 5.0
    assert formats == [Format.Banner, Format.Carousel]

    # Test 2: MaxOccurrences constraint - Enforce max 0 total Banner occurrences
    constraints_2 = [
        LayoutConstraint.max_occurrences(Format.Banner, 0)
    ]
    val, formats = optimize_layout(trays, 8, constraints=constraints_2)
    assert val == 3.5
    assert formats == [Format.Carousel, Format.Carousel]

    # Test 3: NoConsecutive constraint - Disallow sequential Carousel
    constraints_3 = [
        LayoutConstraint.no_consecutive(Format.Carousel)
    ]
    val, formats = optimize_layout(trays, 8, constraints=constraints_3)
    # Carousel + Carousel is disallowed, so we should get Banner + Banner -> 3.0 + 4.5 = 7.5
    assert val == 7.5
    assert formats == [Format.Banner, Format.Banner]
