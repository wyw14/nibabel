# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for lazy BrainModelAxis alignment (``nibabel.cifti2.cifti2_align``)"""

from copy import deepcopy

import numpy as np
import pytest

from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2_align import (
    MISSING_INDEX,
    BrainModelAlignmentError,
    BrainModelAxisProxy,
    align_brain_model_axis,
    compute_brain_model_mapping,
)
from nibabel.tmpdirs import InTemporaryDirectory

from ...testing import assert_array_equal

AFFINE = np.array(
    [
        [-1.5, 0, 0, 90],
        [0, 1.5, 0, -85],
        [0, 0, 1.5, -71],
        [0, 0, 0, 1.0],
    ]
)
VOL_SHAPE = (4, 5, 6)
NVERT_LEFT = 6
NVERT_RIGHT = 5


def surface_axis(vertices, nvertex, name):
    return ci.BrainModelAxis.from_surface(vertices, nvertex, name=name)


def volume_axis(voxels, name):
    return ci.BrainModelAxis(name, voxel=np.asarray(voxels), affine=AFFINE, volume_shape=VOL_SHAPE)


@pytest.fixture
def source_axis():
    # Mixed surface/volume axis, with the left cortex split into two
    # non-contiguous fragments separated by a volume structure.
    return (
        surface_axis([0, 2, 4], NVERT_LEFT, 'CortexLeft')
        + volume_axis([(0, 0, 0), (1, 1, 1), (2, 2, 2)], 'ThalamusLeft')
        + surface_axis([1, 3], NVERT_LEFT, 'CortexLeft')
        + volume_axis([(3, 3, 3)], 'BrainStem')
        + surface_axis([1, 3], NVERT_RIGHT, 'CortexRight')
    )


@pytest.fixture
def target_axis():
    # Reordered subset, reversed surface fragment and elements missing from
    # the source (cortex vertex 5 and voxel (0, 0, 1)).
    return (
        surface_axis([4, 0, 5], NVERT_LEFT, 'CortexLeft')
        + volume_axis([(3, 3, 3), (0, 0, 1)], 'BrainStem')
        + surface_axis([3], NVERT_RIGHT, 'CortexRight')
        + volume_axis([(1, 1, 1)], 'ThalamusLeft')
    )


@pytest.fixture
def source_data(source_axis):
    rng = np.random.default_rng(42)
    return rng.integers(-100, 100, size=(4, len(source_axis)), dtype=np.int32)


@pytest.fixture
def source_image(source_axis, source_data):
    series = ci.SeriesAxis(start=0, step=0.1, size=4)
    return ci.Cifti2Image(source_data, (series, source_axis))


def reference_output(data, mapping, aligned_axis, fill_value):
    """Explicit, non-lazy reference implementation of the aligned output"""
    out_shape = list(data.shape)
    out_shape[aligned_axis] = mapping.size
    has_missing = bool(np.any(mapping == MISSING_INDEX))
    dtype = np.result_type(data.dtype, np.asarray(fill_value).dtype) if has_missing else data.dtype
    out = np.full(out_shape, fill_value, dtype=dtype)
    for target_index, source_index in enumerate(mapping):
        if source_index == MISSING_INDEX:
            continue
        source_selector = [slice(None)] * data.ndim
        output_selector = [slice(None)] * data.ndim
        source_selector[aligned_axis] = int(source_index)
        output_selector[aligned_axis] = target_index
        out[tuple(output_selector)] = data[tuple(source_selector)]
    return out


class RecordingData:
    """Array-like stand-in that records every slicer applied to it"""

    is_proxy = True

    def __init__(self, array):
        self._array = array
        self.accesses = []

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    def __getitem__(self, slicer):
        self.accesses.append(slicer)
        return self._array[slicer]

    def __array__(self, dtype=None):
        self.accesses.append(())
        return np.asarray(self._array, dtype=dtype)


def test_compute_mapping_identities(source_axis, target_axis):
    mapping, missing = compute_brain_model_mapping(source_axis, target_axis)

    # Source layout (positions):
    # 0-2  CortexLeft vertices 0, 2, 4
    # 3-5  ThalamusLeft voxels (0,0,0), (1,1,1), (2,2,2)
    # 6-7  CortexLeft vertices 1, 3 (non-contiguous fragment)
    # 8    BrainStem voxel (3,3,3)
    # 9-10 CortexRight vertices 1, 3
    expected = np.array(
        [
            2,  # cortex left vertex 4
            0,  # cortex left vertex 0
            -1,  # cortex left vertex 5 not in source
            8,  # brain stem voxel (3, 3, 3)
            -1,  # voxel (0, 0, 1) not in source
            10,  # cortex right vertex 3
            4,  # thalamus voxel (1, 1, 1)
        ],
        dtype=np.intp,
    )
    assert mapping.dtype == np.intp
    assert_array_equal(mapping, expected)
    assert [index for index, _ in missing] == [2, 4]


def test_mapping_matches_manual_lookup(source_axis, target_axis):
    mapping, _ = compute_brain_model_mapping(source_axis, target_axis)
    source_surface = source_axis.surface_mask
    target_surface = target_axis.surface_mask
    lookup = {}
    for i in range(len(source_axis)):
        if source_surface[i]:
            key = (True, str(source_axis.name[i]), int(source_axis.vertex[i]))
        else:
            key = (False, *map(int, source_axis.voxel[i]))
        lookup[key] = i
    for j in range(len(target_axis)):
        if target_surface[j]:
            key = (True, str(target_axis.name[j]), int(target_axis.vertex[j]))
        else:
            key = (False, *map(int, target_axis.voxel[j]))
        assert mapping[j] == lookup.get(key, MISSING_INDEX)


def test_align_full_array_matches_reference(source_image, source_data, target_axis):
    aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=-9
    )
    mapping, _ = compute_brain_model_mapping(source_image.header.get_axis(1), target_axis)
    expected = reference_output(source_data, mapping, 1, -9)
    assert aligned.shape == expected.shape
    assert aligned.dataobj.is_proxy is True
    assert_array_equal(np.asarray(aligned.dataobj), expected)
    # Header describes the target brain model, not the source one
    assert aligned.header.get_axis(1) == target_axis
    assert aligned.header.get_axis(0) == source_image.header.get_axis(0)


def test_align_axis_zero(source_axis):
    scalar = ci.ScalarAxis(['map_a', 'map_b', 'map_c'])
    rng = np.random.default_rng(0)
    data = rng.integers(0, 50, size=(len(source_axis), 3), dtype=np.int32)
    image = ci.Cifti2Image(data, (source_axis, scalar))

    target = surface_axis([3, 1], NVERT_RIGHT, 'CortexRight') + volume_axis(
        [(2, 2, 2), (0, 0, 0)], 'ThalamusLeft'
    )
    aligned = align_brain_model_axis(image, target, axis=0)
    mapping, _ = compute_brain_model_mapping(source_axis, target)
    assert_array_equal(np.asarray(aligned.dataobj), reference_output(data, mapping, 0, 0))
    assert aligned.header.get_axis(0) == target
    assert aligned.header.get_axis(1) == scalar
    assert aligned.shape == (len(target), 3)


def test_missing_error_policy(source_image, target_axis):
    with pytest.raises(BrainModelAlignmentError, match='not present in the source'):
        align_brain_model_axis(source_image, target_axis, axis=1)
    # ValueError base class remains usable
    with pytest.raises(ValueError):
        align_brain_model_axis(source_image, target_axis, axis=1)
    with pytest.raises(ValueError, match="missing must be 'error' or 'fill'"):
        align_brain_model_axis(source_image, target_axis, axis=1, missing='nope')


def test_missing_fill_value_and_minus_one_not_last_column(source_image, source_data, target_axis):
    fill_value = -1234
    aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=fill_value
    )
    proxy = aligned.dataobj
    # Missing positions are the first cortex vertex 5 and voxel (0, 0, 1)
    assert_array_equal(proxy.mapping[[2, 4]], [-1, -1])
    result = np.asarray(proxy)
    # -1 must be treated as missing, not aliasing the last source column
    assert_array_equal(result[:, 2], np.full(4, fill_value))
    assert_array_equal(result[:, 4], np.full(4, fill_value))
    assert np.any(source_data[:, -1] != fill_value)
    # Present positions still carry source data
    assert_array_equal(result[:, 0], source_data[:, 2])
    assert_array_equal(result[:, 3], source_data[:, 8])

    # Fill value participates in dtype promotion
    float_aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=np.nan
    )
    assert float_aligned.dataobj.dtype == np.dtype(np.float64)
    float_result = np.asarray(float_aligned.dataobj)
    assert np.all(np.isnan(float_result[:, [2, 4]]))

    half_aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=np.float32(1.5)
    )
    assert half_aligned.dataobj.dtype == np.dtype(np.float64)


def test_duplicate_identities_rejected_before_reading(source_axis, source_data):
    recording = RecordingData(source_data)
    image = ci.Cifti2Image(recording, (ci.SeriesAxis(0, 0.1, 4), source_axis))
    # Duplicate surface identity within the target
    duplicate_target = surface_axis([0, 0], NVERT_LEFT, 'CortexLeft')
    with pytest.raises(BrainModelAlignmentError, match='Duplicate gray matter identity'):
        align_brain_model_axis(image, duplicate_target, axis=1)
    assert recording.accesses == []

    # Duplicate identity within the source axis is rejected as well
    duplicate_source = surface_axis([1, 1], NVERT_RIGHT, 'CortexRight')
    good_target = surface_axis([1], NVERT_RIGHT, 'CortexRight')
    bad_image = ci.Cifti2Image(source_data[:, :2], (ci.SeriesAxis(0, 0.1, 4), duplicate_source))
    with pytest.raises(BrainModelAlignmentError, match='Duplicate gray matter identity'):
        align_brain_model_axis(bad_image, good_target, axis=1)


def test_voxel_structure_conflict_raises():
    conflict = ci.BrainModelAxis(
        ['CIFTI_STRUCTURE_THALAMUS_LEFT', 'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT'],
        voxel=[[0, 0, 0], [0, 0, 0]],
        affine=AFFINE,
        volume_shape=VOL_SHAPE,
    )
    target = volume_axis([(0, 0, 0)], 'ThalamusLeft')
    with pytest.raises(BrainModelAlignmentError, match='conflicting brain structures'):
        compute_brain_model_mapping(conflict, target)
    with pytest.raises(BrainModelAlignmentError, match='conflicting brain structures'):
        compute_brain_model_mapping(target, conflict)


def test_volume_grid_mismatch(source_axis):
    target = volume_axis([(0, 0, 0)], 'ThalamusLeft')

    different_shape = ci.BrainModelAxis(
        'CIFTI_STRUCTURE_THALAMUS_LEFT',
        voxel=[[0, 0, 0]],
        affine=AFFINE,
        volume_shape=(4, 5, 7),
    )
    with pytest.raises(BrainModelAlignmentError, match='Volume shapes differ'):
        compute_brain_model_mapping(source_axis, different_shape)

    different_affine = ci.BrainModelAxis(
        'CIFTI_STRUCTURE_THALAMUS_LEFT',
        voxel=[[0, 0, 0]],
        affine=AFFINE + 10.0,
        volume_shape=VOL_SHAPE,
    )
    with pytest.raises(BrainModelAlignmentError, match='affines differ'):
        compute_brain_model_mapping(source_axis, different_affine)

    # Affine differences within the explicit tolerance are accepted
    close_affine = ci.BrainModelAxis(
        'CIFTI_STRUCTURE_THALAMUS_LEFT',
        voxel=[[0, 0, 0]],
        affine=AFFINE + 1e-8,
        volume_shape=VOL_SHAPE,
    )
    mapping, missing = compute_brain_model_mapping(source_axis, close_affine)
    assert mapping.tolist() == [3]
    assert missing == []

    # Grid checks can be tightened
    with pytest.raises(BrainModelAlignmentError, match='affines differ'):
        compute_brain_model_mapping(source_axis, close_affine, atol=1e-12, rtol=0)

    # A surface-only axis does not trigger volume grid checks
    surface_only = surface_axis([0, 2], NVERT_LEFT, 'CortexLeft')
    mapping, _ = compute_brain_model_mapping(surface_only, target)
    assert mapping.tolist() == [-1]


def test_surface_nvertices_mismatch_rejects():
    source = surface_axis([0, 1], NVERT_LEFT, 'CortexLeft')
    target = surface_axis([0], NVERT_LEFT + 1, 'CortexLeft')
    with pytest.raises(BrainModelAlignmentError, match='vertices in the source axis'):
        compute_brain_model_mapping(source, target)


@pytest.mark.parametrize(
    'slicer',
    [
        (slice(None), slice(None)),
        (slice(None), 1),
        (slice(None), -1),
        (-2, slice(None)),
        (slice(None), slice(0, 4)),
        (slice(None), slice(None, None, 2)),
        (slice(None), slice(None, None, -1)),
        (slice(None, None, -1), slice(None, None, -1)),
        (slice(1, 3), slice(2, 6, 2)),
        (Ellipsis, slice(-2, None)),
        (np.newaxis, slice(None), slice(None)),
        (slice(None), np.newaxis, slice(None)),
        (0, 0),
        (-1, -1),
    ],
)
def test_proxy_slicing_matches_reference(source_image, source_data, target_axis, slicer):
    aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=-9
    )
    mapping, _ = compute_brain_model_mapping(source_image.header.get_axis(1), target_axis)
    expected = reference_output(source_data, mapping, 1, -9)
    assert_array_equal(aligned.dataobj[slicer], expected[slicer])


def test_proxy_fancy_indexing_rejected(source_image, target_axis):
    aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=-9
    )
    with pytest.raises(ValueError, match='basic slicing'):
        aligned.dataobj[np.array([0, 2])]


def test_proxy_contiguous_source_ranges(source_axis, source_data, target_axis):
    recording = RecordingData(source_data)
    image = ci.Cifti2Image(recording, (ci.SeriesAxis(0, 0.1, 4), source_axis))
    aligned = align_brain_model_axis(image, target_axis, axis=1, missing='fill', fill_value=-9)
    proxy = aligned.dataobj

    # Targets 0 and 1 map to source cortex positions 2 and 0: two singleton
    # ranges, each read as its own contiguous slice.
    recording.accesses.clear()
    proxy[:, 0:2]
    aligned_slicers = [access[1] for access in recording.accesses]
    assert aligned_slicers == [slice(0, 1, None), slice(2, 3, None)]
    # Other dimensions are passed through unchanged as full basic slices.
    assert all(access[0] == slice(None) for access in recording.accesses)

    # A contiguous run of required source indices is read in one slice.
    contiguous_target = volume_axis([(0, 0, 0), (1, 1, 1), (2, 2, 2)], 'ThalamusLeft')
    contiguous_aligned = align_brain_model_axis(image, contiguous_target, axis=1)
    recording.accesses.clear()
    contiguous_aligned.dataobj[:, :]
    assert len(recording.accesses) == 1
    assert recording.accesses[0][1] == slice(3, 6, None)

    # Small slice touches only the corresponding source range, not the matrix.
    recording.accesses.clear()
    contiguous_aligned.dataobj[1:3, 1:2]
    assert len(recording.accesses) == 1
    assert recording.accesses[0] == (slice(1, 3, None), slice(4, 5, None))

    # Negative integer indexing becomes a positive integer source access.
    recording.accesses.clear()
    contiguous_aligned.dataobj[:, -1]
    assert recording.accesses == [(slice(None), 5)]

    # Reverse order: needed source indices are still contiguous after sorting,
    # so a single contiguous read is issued and target order is restored on
    # scatter.
    reversed_target = volume_axis([(2, 2, 2), (1, 1, 1), (0, 0, 0)], 'ThalamusLeft')
    reversed_aligned = align_brain_model_axis(image, reversed_target, axis=1)
    recording.accesses.clear()
    reversed_result = reversed_aligned.dataobj[()]
    assert [access[1] for access in recording.accesses] == [slice(3, 6, None)]
    mapping, _ = compute_brain_model_mapping(source_axis, reversed_target)
    assert_array_equal(reversed_result, reference_output(source_data, mapping, 1, 0))
    # And the requested (reversed) order is actually restored.
    assert_array_equal(reversed_result[0, :], source_data[0, [5, 4, 3]])


def test_empty_and_all_missing_outputs_do_not_read_source(source_axis, source_data, target_axis):
    recording = RecordingData(source_data)
    image = ci.Cifti2Image(recording, (ci.SeriesAxis(0, 0.1, 4), source_axis))
    aligned = align_brain_model_axis(image, target_axis, axis=1, missing='fill', fill_value=-9)

    recording.accesses.clear()
    empty = aligned.dataobj[:, slice(0, 0)]
    assert empty.shape == (4, 0)
    assert empty.dtype == aligned.dataobj.dtype
    assert recording.accesses == []

    # Targets 2 and 4 (stride 2) are exactly the missing elements.
    recording.accesses.clear()
    all_missing = aligned.dataobj[:, 2:5:2]
    assert recording.accesses == []
    assert_array_equal(all_missing, np.full((4, 2), -9, dtype=np.int32))

    # Scalar selection of a missing brainordinate also avoids the source.
    recording.accesses.clear()
    scalar_missing = aligned.dataobj[:, 2]
    assert recording.accesses == []
    assert_array_equal(scalar_missing, np.full(4, -9, dtype=np.int32))

    # A target made exclusively of missing elements is filled end to end
    # without a single source read, including a full-array conversion.
    only_missing = surface_axis([5], NVERT_LEFT, 'CortexLeft')
    missing_aligned = align_brain_model_axis(
        image, only_missing, axis=1, missing='fill', fill_value=np.nan
    )
    recording.accesses.clear()
    full = np.asarray(missing_aligned.dataobj)
    assert recording.accesses == []
    assert full.shape == (4, 1)
    assert np.all(np.isnan(full))


def test_axes_and_mapping_are_snapshots(source_image, target_axis):
    target_copy = deepcopy(target_axis)
    aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=-9
    )
    mapping, _ = compute_brain_model_mapping(source_image.header.get_axis(1), target_copy)
    expected = reference_output(source_image.dataobj, mapping, 1, -9)

    # Mutate the caller-owned target axis after construction.
    target_axis.name[:] = 'CIFTI_STRUCTURE_BRAIN_STEM'
    target_axis.vertex[0] = 999
    target_axis.voxel[0] = [9, 9, 9]
    # Mutate the source axis arrays as well.
    source_bm = source_image.header.get_axis(1)
    source_bm.vertex[0] = 999
    source_bm.voxel[0] = [8, 8, 8]

    assert_array_equal(np.asarray(aligned.dataobj), expected)
    aligned_bm = aligned.header.get_axis(1)
    assert aligned_bm == target_copy
    assert_array_equal(aligned_bm.name, target_copy.name)
    assert_array_equal(aligned_bm.vertex, target_copy.vertex)
    assert_array_equal(aligned_bm.voxel, target_copy.voxel)

    # The mapping stored in the proxy is independent of the array passed in.
    external_mapping = np.array([0, 2, -1], dtype=np.intp)
    proxy = BrainModelAxisProxy(source_image.dataobj, external_mapping, 1, fill_value=-9)
    external_mapping[0] = -1
    assert proxy.mapping[0] == 0
    assert not proxy.mapping.flags.writeable


def test_source_image_cache_is_not_filled(source_axis, source_data, target_axis):
    with InTemporaryDirectory():
        series = ci.SeriesAxis(start=0, step=0.1, size=4)
        ci.save(ci.Cifti2Image(source_data, (series, source_axis)), 'source.dscalar.nii')
        import nibabel as nib

        loaded = nib.load('source.dscalar.nii')
        assert loaded.dataobj.is_proxy
        aligned = align_brain_model_axis(
            loaded, target_axis, axis=1, missing='fill', fill_value=-9
        )
        _ = aligned.dataobj[:, 0:3]
        assert loaded._fdata_cache is None
        assert loaded._data_cache is None
        assert loaded.in_memory is False


def test_nifti_header_dims_intent_and_metadata(source_image, target_axis):
    source_image.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS')
    aligned = align_brain_model_axis(
        source_image, target_axis, axis=1, missing='fill', fill_value=0
    )
    # Outer NIfTI-2 dimensions reserve the first four slots.
    assert aligned.nifti_header.get_data_shape() == (1, 1, 1, 1) + aligned.shape
    assert aligned.nifti_header.get_intent()[0] == 'ConnDenseScalar'
    # The series metadata on the other matrix axis is preserved.
    assert aligned.header.get_axis(0) == source_image.header.get_axis(0)


def test_save_and_reload_roundtrip(source_axis, source_data, target_axis):
    series = ci.SeriesAxis(start=0, step=0.1, size=4)
    image = ci.Cifti2Image(source_data, (series, source_axis))
    image.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS')
    aligned = align_brain_model_axis(image, target_axis, axis=1, missing='fill', fill_value=-9)
    mapping, _ = compute_brain_model_mapping(source_axis, target_axis)
    expected = reference_output(source_data, mapping, 1, -9)

    with InTemporaryDirectory():
        ci.save(aligned, 'aligned.dscalar.nii')
        import nibabel as nib

        reloaded = nib.load('aligned.dscalar.nii')
        assert reloaded.dataobj.is_proxy
        assert reloaded.shape == aligned.shape
        assert reloaded.header.get_axis(1) == target_axis
        assert reloaded.header.get_axis(0) == series
        assert reloaded.nifti_header.get_intent()[0] == 'ConnDenseScalar'
        assert reloaded.nifti_header.get_data_shape() == (1, 1, 1, 1) + aligned.shape
        assert_array_equal(np.asanyarray(reloaded.dataobj), expected)
        # Storage dtype follows the proxy dtype
        assert reloaded.nifti_header.get_data_dtype() == aligned.dataobj.dtype


def test_reversed_full_alignment(source_axis, source_data):
    target = source_axis[slice(None, None, -1)]
    image = ci.Cifti2Image(source_data, (ci.SeriesAxis(0, 0.1, 4), source_axis))
    aligned = align_brain_model_axis(image, target, axis=1)
    mapping, missing = compute_brain_model_mapping(source_axis, target)
    assert missing == []
    assert_array_equal(np.asarray(aligned.dataobj), source_data[:, ::-1])
    assert_array_equal(mapping, np.arange(len(source_axis))[::-1])


def test_non_brain_model_axis_rejected(source_axis):
    scalar = ci.ScalarAxis(['a', 'b'])
    image = ci.Cifti2Image(np.zeros((2, len(source_axis))), (scalar, source_axis))
    with pytest.raises(BrainModelAlignmentError, match='not a BrainModelAxis'):
        align_brain_model_axis(image, source_axis, axis=0)
    with pytest.raises(TypeError):
        align_brain_model_axis(np.zeros((2, 2)), source_axis)
    with pytest.raises(TypeError):
        align_brain_model_axis(image, scalar, axis=1)
