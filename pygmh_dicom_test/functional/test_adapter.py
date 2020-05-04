
import pytest

from pygmh_dicom.adapter import Adapter
from pygmh_dicom_test.assets import asset_path


def test_load_single_volume():

    adapter = Adapter()

    image = adapter.read(
        asset_path("single_volume")
    )

    assert image.get_image_data().shape == (42, 512, 512)
    assert image.get_voxel_spacing() == (5.0, 0.76953125, 0.76953125)


def test_inconsistent_zdiff():

    adapter = Adapter()

    with pytest.raises(Exception):

        adapter.read(
            asset_path("inconsistent_zdiff")
        )


def test_missing_slices():

    adapter = Adapter()

    with pytest.raises(Exception):

        adapter.read(
            asset_path("missing_slices")
        )


def test_series_discovery():

    adapter = Adapter()

    series_uids = adapter.discover_series(
        asset_path("single_volume")
    )

    assert series_uids == {
        "1.3.6.1.4.1.14519.5.2.1.7009.2403.946109214067775408970852594736"
    }


def test_discover_series_recursive():

    adapter = Adapter()

    series_uids = adapter.discover_series_recursive(
        asset_path("")
    )

    assert len({
        "1.3.6.1.4.1.14519.5.2.1.7009.2403.946109214067775408970852594736",
        "1.2.276.0.50.192168001099.8252157.14547392.106",
        "1.2.276.0.50.192168001099.8252157.14547392.391",
    }.difference(set(series_uids.keys()))) == 0
