def test_get_valid_properties_filters_known_quantities():
    from gfs_dynamical_core._property_utils import get_valid_properties
    gfs_props = {"air_temperature": {"dims": ["mid_levels", "lat", "lon"]}}
    prognostic = {
        "air_temperature": {"dims": ["mid_levels", "lat", "lon"], "units": "K"},
        "my_forcing": {"dims": ["mid_levels", "lat", "lon"], "units": "K s^-1"},
    }
    result = get_valid_properties(gfs_props, prognostic, "input")
    assert "air_temperature" not in result   # already known to the core
    assert "my_forcing" in result            # extra quantity surfaced
