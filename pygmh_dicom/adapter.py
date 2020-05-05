
import glob
import logging
import os
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
import pydicom

from pygmh.model import MetaData, Image, Vector3
from pygmh.persistence.interface import IAdapter


DICOM_META_DATA_KEY = "dicom"


class Adapter(IAdapter):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def read(self, path: str) -> Image:

        self._logger.info("Reading single dicom image/series from: " + path)

        images = self.read_multiple(path)

        assert len(images) == 1, "Expected single dicom image in path {}. Got {}: {}".format(
            path,
            len(images),
            ", ".join([
                image.get_identifier() or "unknown"
                for image in images
            ])
        )

        return images[0]

    def read_multiple(self, path: str) -> List[Image]:
        """
        Allows to read in multiple images at once in case they are mixed within a single directory.
        """

        self._logger.info("Reading dicom images/series from: " + path)

        file_paths_by_series_uid = self._build_file_paths_by_series_uid_map(path)

        self._logger.info(
            "Reading in {} images with series UIDs: {}".format(
                len(file_paths_by_series_uid),
                ", ".join([uid for uid in file_paths_by_series_uid.keys()])
            )
        )

        images: List[Image] = list()
        for uid, file_path_list in file_paths_by_series_uid.items():

            self._logger.debug("Reading in series {}".format(uid))

            # read in all slices
            slices = [
                pydicom.dcmread(file_path)
                for file_path in file_path_list
            ]

            # filter non-image slices
            # todo: validate criterion
            slices = [
                slice
                for slice in slices
                if "ImagePositionPatient" in slice
            ]
            if len(slices) == 0:
                continue

            # load image volume and meta-data
            volume = self._build_volume(slices)
            image_meta_data, slice_meta_data_by_index = self._get_meta_data(slices)

            image = Image(image_data=volume)
            image.get_meta_data().update({
                DICOM_META_DATA_KEY: image_meta_data
            })

            # attach slice-specific meta-data
            for index, slice_meta_data in slice_meta_data_by_index.items():

                image.get_or_add_slice(index).get_meta_data().update({
                    DICOM_META_DATA_KEY: slice_meta_data
                })

            # deduce image information from dicom meta-data
            image.set_voxel_size(self._get_voxel_size(image))
            image.set_voxel_spacing(self._get_voxel_spacing(image))

            images.append(image)

        return images

    def write(self, image: Image, path: str) -> None:
        raise NotImplementedError()

    def discover_series(self, path: str) -> Set[str]:
        """Returns a set of DICOM series UIDs that are present in the given directory."""

        return set(self._build_file_paths_by_series_uid_map(path).keys())

    def discover_series_recursive(self, path: str) -> Dict[str, str]:

        assert os.path.isdir(path), "Given path is not a directory: " + path

        result = {}

        for path in glob.glob(os.path.join(path, "**/"), recursive=True):

            discovered_series = self.discover_series(path)

            # nothing found in this sub-dir
            if len(discovered_series) == 0:
                continue

            for series in discovered_series:

                if series in result.keys():
                    raise Exception("Dicom series UID collision: {} is present both in {} and {}".format(
                        series, result[series], path
                    ))

                result[series] = path

        return result

    def _get_voxel_size(self, image: Image) -> Optional[Vector3]:
        """Deduces the voxel size based on slice thickness and pixel spacing.

        Relies on the assumption of non-overlap and non-sparseness in the xy-plane.
        """

        pixel_spacing = self._get_pixel_spacing(image)

        if pixel_spacing is None:
            return None

        try:
            slice_thickness = image.get_meta_data()["Slice Thickness"]
            assert isinstance(slice_thickness, float)
            assert slice_thickness > 0
        except KeyError:
            return None

        return (
            slice_thickness,
            pixel_spacing[1],
            pixel_spacing[0],
        )

    def _get_voxel_spacing(self, image: Image) -> Optional[Vector3]:
        """Deduces the voxel spacing based on slice increment and pixel spacing."""

        pixel_spacing = self._get_pixel_spacing(image)

        if pixel_spacing is None:
            return None

        explicit_slice_increment: Optional[float] = None

        # Read explicit value from "Spacing Between Slices" (0018,0088)
        try:
            explicit_slice_increment = image.get_meta_data()[DICOM_META_DATA_KEY][str(0x00180088)]
            assert isinstance(explicit_slice_increment, float)
            explicit_slice_increment = abs(explicit_slice_increment)
        except KeyError:
            pass

        implicit_slice_increment = self._calculate_z_spacing(image)

        if explicit_slice_increment is None and implicit_slice_increment is None:
            # Cannot do anything
            return None

        elif explicit_slice_increment is not None and implicit_slice_increment is not None:
            # Assert consistency
            assert explicit_slice_increment == implicit_slice_increment,\
                "Derived slice increment differs from defined value"

        slice_increment = explicit_slice_increment if explicit_slice_increment is not None else implicit_slice_increment

        return (
            slice_increment,
            pixel_spacing[1],
            pixel_spacing[0],
        )

    def _get_pixel_spacing(self, image: Image) -> Optional[Tuple[float, float]]:

        try:
            # "Pixel Spacing" (0028,0030)
            pixel_spacing = image.get_meta_data()[DICOM_META_DATA_KEY][str(0x00280030)]
        except KeyError:
            return None

        assert isinstance(pixel_spacing, list)
        assert len(pixel_spacing) == 2
        assert all(isinstance(value, float) for value in pixel_spacing)

        return (
            pixel_spacing[0],
            pixel_spacing[1]
        )

    def _calculate_z_spacing(self, image: Image) -> Optional[float]:

        ordered_slices = image.get_ordered_slices()

        # Cannot calculate spacing with less then two slices
        if len(ordered_slices) < 2:
            return None

        try:
            # "Slice Location" (0020,1041)
            location1 = ordered_slices[0].get_meta_data()[DICOM_META_DATA_KEY][str(0x00201041)]
            location2 = ordered_slices[1].get_meta_data()[DICOM_META_DATA_KEY][str(0x00201041)]
        except KeyError:
            return None

        assert isinstance(location1, float)
        assert isinstance(location2, float)

        return abs(location1 - location2)

    def _get_meta_data(self, slices: List[pydicom.dataset.FileDataset]) -> (MetaData, Dict[int, MetaData]):
        """Reads all meta data from the given set of dicom slice data and returns two sets with consistent and slice-
        specific meta-data respectively."""

        slice_meta_data_dicts = [
            self._read_slice_meta_data(current_slice)
            for current_slice in slices
        ]

        _, keys_with_consistent_data, keys_with_different_data = self._get_meta_data_key_sets(slice_meta_data_dicts)

        # deduce image meta-data by using everything that is consistent across all slices
        image_meta_data = MetaData()
        arbitrary_slice_meta_data_dict = slice_meta_data_dicts[0]
        for key in keys_with_consistent_data:
            image_meta_data[key] = arbitrary_slice_meta_data_dict[key]

        # deduce slice-specific meta-data by using all keys with different data
        slice_meta_data_by_index = dict()
        for index, slice_meta_data_dict in enumerate(slice_meta_data_dicts):
            slice_meta_data = MetaData({
                key: slice_meta_data_dict[key]
                for key in keys_with_different_data
                if key in slice_meta_data_dict
            })
            if bool(slice_meta_data):
                slice_meta_data_by_index[index] = slice_meta_data

        return image_meta_data, slice_meta_data_by_index

    def _get_meta_data_key_sets(self, slice_meta_data_dicts: List[dict]) -> (set, set, set):
        """ Builds two distinct sets of keys containing:
        1) names that have consistent data across all slices
        2) names that have different data for at least one slice
        """

        keys_with_consistent_data = set()
        keys_with_different_data = set()
        for slice_meta_data_dict in slice_meta_data_dicts:
            for key in slice_meta_data_dict:

                is_unique = False

                for cmp_slice_meta_data in slice_meta_data_dicts:
                    if key not in cmp_slice_meta_data or slice_meta_data_dict[key] != cmp_slice_meta_data[key]:
                        is_unique = True
                        break

                if is_unique:
                    keys_with_different_data.add(key)
                else:
                    keys_with_consistent_data.add(key)

        assert not keys_with_consistent_data.intersection(keys_with_different_data)

        all_keys = set()
        all_keys.union(keys_with_consistent_data)
        all_keys.union(keys_with_different_data)

        return all_keys, keys_with_consistent_data, keys_with_different_data

    def _read_slice_meta_data(self, slice: pydicom.FileDataset) -> dict:
        """Reads in the meta-data from the given slice and parses them into plain python data-types."""

        meta_data_dict = dict()

        # iterate over all meta data fields
        for element in slice:

            assert element.tag not in meta_data_dict,\
                "Duplicate DICOM header tag: {} {}".format(element.name, element.tag)

            meta_data_dict[str(int(element.tag))] = self._format_slice_meta_data_value(element.name, element.value)

        return meta_data_dict

    def _format_slice_meta_data_value(self, name: str, value):

        # convert sequences to string representation.
        # Needs to be before the if statement for MultiValue (as sequence is a subclass of Multivalue)
        if isinstance(value, pydicom.sequence.Sequence):
            value = value.__str__()

        # convert multivalues to lists
        elif isinstance(value, pydicom.multival.MultiValue):
            value = [
                self._format_slice_meta_data_value(name, val)
                for val in list(value)
            ]

        # convert PersonNames to tuple then transform to str
        elif isinstance(value, pydicom.valuerep.PersonName3):
            value = "".join(value.components)

        elif isinstance(value, pydicom.valuerep.DSfloat):
            value = float(value)

        elif isinstance(value, bytes):

            # cannot be processed
            value = None

        if name in [
            "PatientBirthDate",
            "PatientBirthTime",
            "StudyDate",
            "StudyTime",
            "SeriesTime",
            "SeriesDate",
            "SeriesNumber",
        ]:
            value = int(value)

        return value

    def _build_volume(self, slices: List[pydicom.dataset.FileDataset]) -> np.ndarray:
        """Builds the image volume by the given set of slices."""

        # sort slices within volume
        assert len(slices) > 0
        slices.sort(key=lambda current_slice: self._get_slice_location(current_slice))

        # todo: drop assertion and fill with empty data to support potential use-cases using only some slices
        self._assert_consistent_slice_distances(slices)

        # read in slice images
        slice_images = list()
        for current_slice in slices:

            slice_image = current_slice.pixel_array

            # rescale pixel values
            if current_slice.RescaleSlope or current_slice.RescaleIntercept:
                assert current_slice.RescaleSlope is not None and current_slice.RescaleIntercept is not None,\
                    "Only one information of scaling-slope and -intercept is present"
                slice_image = slice_image * current_slice.RescaleSlope + current_slice.RescaleIntercept

            # type-cast
            slice_image = slice_image.astype(np.int32)

            slice_images.append(slice_image)

        self._assert_consistent_slice_dimensions(slice_images)

        # stack slices to get full volume
        volume = np.stack(slice_images)

        volume = np.flip(volume, 1)

        return volume

    def _assert_consistent_slice_distances(self, slices: List[pydicom.dataset.FileDataset]):

        if len(slices) < 2:
            return

        reference_distance = self._get_slice_location(slices[0]) - self._get_slice_location(slices[1])

        assert abs(reference_distance) > 0.001, "No distance in slice locations. (Orientation?)"

        for i in range(len(slices) - 1):

            current_distance = self._get_slice_location(slices[i]) - self._get_slice_location(slices[i + 1])

            assert abs(current_distance - reference_distance) < 0.001, \
                "Distance between slices are not consistent"

    def _assert_consistent_slice_dimensions(self, slice_images: List[np.ndarray]):

        if len(slice_images) < 1:
            return

        reference_dimensions = slice_images[0].shape

        for slice_image in slice_images:

            assert slice_image.shape == reference_dimensions, "Slice dimension mismatch"

    def _get_slice_location(self, slice: pydicom.dataset.FileDataset) -> float:

        z_index = 0 if ("AnatomicalOrientationType" in slice and slice.AnatomicalOrientationType != "BIPED") else 2

        return float(slice.ImagePositionPatient[z_index])

    def _build_file_paths_by_series_uid_map(self, path: str) -> Dict[str, List[str]]:
        """
        Builds dictionary of lists, mapping series uid's to lists of file paths containing slices of that series.
        """

        assert os.path.isdir(path), "Trying to find dicoms in non-directory: " + path

        result = dict()
        for file_path in glob.glob(path + "/*.dcm"):

            # ignore directories
            if not os.path.isfile(file_path):
                continue

            # skip DICOMDIR.dcm file
            if file_path.endswith("DICOMDIR.dcm"):
                continue

            sop = pydicom.dcmread(file_path)

            # todo: assert SOP type indicates a CT/MRT slice and dismiss those that dont (with warning)

            if sop.SeriesInstanceUID not in result:
                result[sop.SeriesInstanceUID] = []

            result[sop.SeriesInstanceUID].append(file_path)

        return result
