# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for the weighted CIFTI-2 parcellation interface"""

import tracemalloc

import numpy as np
import pytest

import nibabel as nib
from nibabel.cifti2 import Cifti2Image, ParcelsAxis, ScalarAxis, SeriesAxis, parcellate
from nibabel.cifti2 import cifti2_axes as axes

rng = np.random.default_rng(42)

VOL_SHAPE = (2, 3, 2)
VOL_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
VOXELS = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])


def make_source_brain_model():
    """Brain model with two surface structures and one volume structure"""
    cortex_l = axes.BrainModelAxis.from_surface(np.arange(8), nvertex=10, name='CortexLeft')
    cortex_r = axes.BrainModelAxis.from_surface(np.arange(6), nvertex=8, name='CortexRight')
    thalamus = axes.BrainModelAxis(
        'ThalamusLeft', voxel=VOXELS, affine=VOL_AFFINE, volume_shape=VOL_SHAPE
    )
    return cortex_l + cortex_r + thalamus


def make_parcels():
    """Parcel definitions exercising overlap, shuffled members, zero weights
    and mixed surface/volume membership"""
    # Shuffled surface members; weights are aligned to the members, not positions
    p1_bm = axes.BrainModelAxis.from_surface([3, 1], nvertex=10, name='CortexLeft')
    p1_w = [2.0, 0.5]

    # Overlaps p1 on cortex-left vertex 1; one zero weight; spans two surfaces
    p2_bm = axes.BrainModelAxis.from_surface(
        [1, 5, 6], nvertex=10, name='CortexLeft'
    ) + axes.BrainModelAxis.from_surface([2], nvertex=8, name='CortexRight')
    p2_w = [1.0, 0.0, 3.0, 2.0]

    # Mixed surface and volume parcel, shuffled across structures
    p3_bm = (
        axes.BrainModelAxis.from_surface([4], nvertex=10, name='CortexLeft')
        + axes.BrainModelAxis.from_surface([0], nvertex=8, name='CortexRight')
        + axes.BrainModelAxis(
            'ThalamusLeft',
            voxel=np.array([[1, 1, 1], [0, 0, 0]]),
            affine=VOL_AFFINE,
            volume_shape=VOL_SHAPE,
        )
    )
    p3_w = [1.5, 2.5, 0.0, 4.0]

    # Volume-only parcel
    p4_bm = axes.BrainModelAxis(
        'ThalamusLeft',
        voxel=np.array([[0, 1, 0], [1, 0, 0]]),
        affine=VOL_AFFINE,
        volume_shape=VOL_SHAPE,
    )
    p4_w = [0.7, 1.3]
    return [('p1', p1_bm), ('p2', p2_bm), ('p3', p3_bm), ('p4', p4_bm)], [
        p1_w,
        p2_w,
        p3_w,
        p4_w,
    ]


def reference_parcellate(
    data, src_bm, parcels, weights, axis=-1, nan_policy='propagate', fill_value=np.nan
):
    """Independent, deliberately straightforward per-parcel reference implementation"""
    identity = {}
    for idx in range(src_bm.size):
        if src_bm.name[idx] in src_bm.nvertices:
            identity[('surface', str(src_bm.name[idx]), int(src_bm.vertex[idx]))] = idx
        else:
            identity[('volume', tuple(int(v) for v in src_bm.voxel[idx]))] = idx

    r = axis % data.ndim
    other_shape = tuple(size for dim, size in enumerate(data.shape) if dim != r)
    n_other = int(np.prod(other_shape)) if other_shape else 1
    moved = np.moveaxis(data.astype(np.float64), r, -1).reshape(n_other, data.shape[r])

    out = np.empty((n_other, len(parcels)), dtype=np.float64)
    for p_idx, ((_, bm), parcel_weights) in enumerate(zip(parcels, weights)):
        w = np.asarray(parcel_weights, dtype=np.float64)
        surface_mask = bm.surface_mask
        member_idx = np.empty(bm.size, dtype=np.intp)
        for j in range(bm.size):
            if surface_mask[j]:
                key = ('surface', str(bm.name[j]), int(bm.vertex[j]))
            else:
                key = ('volume', tuple(int(v) for v in bm.voxel[j]))
            member_idx[j] = identity[key]
        values = moved[:, member_idx]
        for row in range(n_other):
            x = values[row]
            if nan_policy == 'propagate':
                if np.any(np.isnan(x[w > 0])):
                    out[row, p_idx] = np.nan
                else:
                    out[row, p_idx] = np.nansum(w * x) / w.sum()
            else:
                valid = ~np.isnan(x)
                denominator = w[valid].sum()
                if denominator == 0:
                    out[row, p_idx] = fill_value
                else:
                    out[row, p_idx] = (w[valid] * x[valid]).sum() / denominator

    moved_shape = other_shape + (len(parcels),)
    return np.ascontiguousarray(np.moveaxis(out.reshape(moved_shape), -1, r))


@pytest.fixture
def source_image():
    src_bm = make_source_brain_model()
    series = SeriesAxis(start=0, step=2, size=7)
    data = rng.normal(size=(7, src_bm.size))
    return Cifti2Image(data, header=[series, src_bm]), data, src_bm, series


@pytest.fixture
def parcel_defs():
    return make_parcels()


def test_parcellate_matches_independent_reference(source_image, parcel_defs):
    img, data, src_bm, series = source_image
    parcels, weights = parcel_defs
    result = parcellate(img, parcels, weights)

    expected = reference_parcellate(data, src_bm, parcels, weights)
    assert isinstance(result, Cifti2Image)
    assert result.shape == (7, 4)
    assert result.dataobj.dtype == np.float64
    np.testing.assert_allclose(np.asarray(result.dataobj), expected)

    # The reduced axis is a ParcelsAxis in the exact input order, the series axis survives
    parcel_axis = result.header.get_axis(1)
    assert isinstance(parcel_axis, ParcelsAxis)
    assert parcel_axis == ParcelsAxis.from_brain_models(parcels)
    assert parcel_axis.name.tolist() == ['p1', 'p2', 'p3', 'p4']
    assert result.header.get_axis(0) == series
    # Outer NIfTI-2 header carries the new CIFTI dimensions
    assert result.nifti_header.get_data_shape() == (1, 1, 1, 1, 7, 4)


def test_overlapping_parcels_are_independent(source_image, parcel_defs):
    img, data, src_bm, _ = source_image
    parcels, weights = parcel_defs
    # p1 and p2 share cortex-left vertex 1; both must get the correct own weighted mean
    result = np.asarray(parcellate(img, parcels, weights).dataobj)
    expected = reference_parcellate(data, src_bm, parcels, weights)
    np.testing.assert_allclose(result, expected)

    p1_only = np.average(data[:, _cl_vertices(src_bm, [3, 1])], weights=[2.0, 0.5], axis=1)
    np.testing.assert_allclose(result[:, 0], p1_only)


def _cl_vertices(src_bm, vertices):
    return [
        int(
            np.where((src_bm.name == 'CIFTI_STRUCTURE_CORTEX_LEFT') & (src_bm.vertex == vertex))[
                0
            ][0]
        )
        for vertex in vertices
    ]


def test_matching_uses_identity_not_member_position(source_image):
    img, data, src_bm, _ = source_image
    # Same members, opposite order; weights travel with their member
    shuffled = [
        ('a', axes.BrainModelAxis.from_surface([3, 1], 10, 'CortexLeft')),
    ]
    ordered = [
        ('a', axes.BrainModelAxis.from_surface([1, 3], 10, 'CortexLeft')),
    ]
    out_shuffled = np.asarray(parcellate(img, shuffled, [[2.0, 0.5]]).dataobj)
    out_ordered = np.asarray(parcellate(img, ordered, [[0.5, 2.0]]).dataobj)
    np.testing.assert_allclose(out_shuffled, out_ordered)
    expected = reference_parcellate(data, src_bm, shuffled, [[2.0, 0.5]])
    np.testing.assert_allclose(out_shuffled, expected)

    # A parcel whose members line up positionally with different source vertices
    # must use the identities, not positional multiplication
    misleading = [
        ('a', axes.BrainModelAxis.from_surface([7, 0], 10, 'CortexLeft')),
    ]
    out = np.asarray(parcellate(img, misleading, [[1.0, 1.0]]).dataobj)
    expected = reference_parcellate(data, src_bm, misleading, [[1.0, 1.0]])
    np.testing.assert_allclose(out, expected)


def test_zero_weights_do_not_pollute_with_nan(source_image, parcel_defs):
    img, data, src_bm, _ = source_image
    parcels, weights = parcel_defs
    data = data.copy()
    # p2's second member (cortex-left vertex 5) has zero weight: poison it with NaN
    cl5 = _cl_vertices(src_bm, [5])[0]
    data[:, cl5] = np.nan
    img = Cifti2Image(data, header=[img.header.get_axis(0), src_bm])

    propagated = np.asarray(parcellate(img, parcels, weights, nan_policy='propagate').dataobj)
    # p2 stays finite because the NaN member carries zero weight; other parcels unaffected
    assert np.isfinite(propagated[:, 1]).all()
    expected = reference_parcellate(data, src_bm, parcels, weights, nan_policy='propagate')
    np.testing.assert_allclose(propagated, expected)

    omitted = np.asarray(parcellate(img, parcels, weights, nan_policy='omit').dataobj)
    np.testing.assert_allclose(omitted, expected)


def test_nan_strategies_propagate_and_omit(source_image, parcel_defs):
    img, data, src_bm, _ = source_image
    parcels, weights = parcel_defs
    data = data.copy()
    # p1 has two positive-weight members; NaN one of them at a single row
    cl3 = _cl_vertices(src_bm, [3])[0]
    data[2, cl3] = np.nan
    img = Cifti2Image(data, header=[img.header.get_axis(0), src_bm])

    propagated = np.asarray(parcellate(img, parcels, weights, nan_policy='propagate').dataobj)
    assert np.isnan(propagated[2, 0])
    assert np.isfinite(propagated[np.arange(7) != 2, 0]).all()
    expected_prop = reference_parcellate(data, src_bm, parcels, weights, nan_policy='propagate')
    np.testing.assert_allclose(propagated, expected_prop)

    omitted = np.asarray(parcellate(img, parcels, weights, nan_policy='omit').dataobj)
    # At row 2 p1 falls back to its single remaining valid member
    cl1 = _cl_vertices(src_bm, [1])[0]
    assert omitted[2, 0] == pytest.approx(data[2, cl1])
    expected_omit = reference_parcellate(data, src_bm, parcels, weights, nan_policy='omit')
    np.testing.assert_allclose(omitted, expected_omit)


def test_omit_returns_fill_value_when_all_missing(source_image):
    img, data, src_bm, _ = source_image
    parcel = [
        ('lonely', axes.BrainModelAxis.from_surface([2, 3], 10, 'CortexLeft')),
    ]
    weights = [[1.0, 0.0]]  # only vertex 2 has positive weight
    data = data.copy()
    cl2 = _cl_vertices(src_bm, [2])[0]
    data[:, cl2] = np.nan  # every position loses its only positive-weight member
    img = Cifti2Image(data, header=[img.header.get_axis(0), src_bm])

    default_out = np.asarray(parcellate(img, parcel, weights, nan_policy='omit').dataobj)
    assert np.isnan(default_out).all()

    filled = np.asarray(
        parcellate(img, parcel, weights, nan_policy='omit', fill_value=-999.0).dataobj
    )
    np.testing.assert_array_equal(filled[:, 0], np.full(7, -999.0))
    # propagate still reports NaN here
    assert np.isnan(np.asarray(parcellate(img, parcel, weights).dataobj)).all()


def test_mixed_surface_and_volume_parcel(source_image, parcel_defs):
    img, data, src_bm, _ = source_image
    parcels, weights = parcel_defs
    result = parcellate(img, parcels, weights)
    parcel_axis = result.header.get_axis(1)

    # p3 records both its surface vertices and its volume voxels exactly
    p3_voxels, p3_vertices = parcel_axis['p3']
    assert set(p3_vertices.keys()) == {
        'CIFTI_STRUCTURE_CORTEX_LEFT',
        'CIFTI_STRUCTURE_CORTEX_RIGHT',
    }
    np.testing.assert_array_equal(p3_vertices['CIFTI_STRUCTURE_CORTEX_LEFT'], [4])
    np.testing.assert_array_equal(p3_vertices['CIFTI_STRUCTURE_CORTEX_RIGHT'], [0])
    np.testing.assert_array_equal(
        np.sort(p3_voxels, axis=0), np.sort([[1, 1, 1], [0, 0, 0]], axis=0)
    )
    assert parcel_axis.affine is not None
    assert parcel_axis.volume_shape == VOL_SHAPE
    assert parcel_axis.nvertices == {
        'CIFTI_STRUCTURE_CORTEX_LEFT': 10,
        'CIFTI_STRUCTURE_CORTEX_RIGHT': 8,
    }
    expected = reference_parcellate(data, src_bm, parcels, weights)
    np.testing.assert_allclose(np.asarray(result.dataobj), expected)


def test_reduction_axis_switch_and_three_dimensions():
    src_bm = make_source_brain_model()
    series = SeriesAxis(start=0, step=1, size=4)
    scalar = ScalarAxis(['map-a', 'map-b', 'map-c'], meta=[{'k': 'v'}, {}, {}])
    data = rng.normal(size=(4, src_bm.size, 3))

    # Brain model on the middle axis
    img_mid = Cifti2Image(data, header=[series, src_bm, scalar])
    out_mid = parcellate(img_mid, *make_parcels(), axis=1)
    assert out_mid.shape == (4, 4, 3)
    assert out_mid.header.get_axis(0) == series
    assert isinstance(out_mid.header.get_axis(1), ParcelsAxis)
    assert out_mid.header.get_axis(2) == scalar
    expected = reference_parcellate(data, src_bm, *make_parcels(), axis=1)
    np.testing.assert_allclose(np.asarray(out_mid.dataobj), expected)

    # Same data with the brain model on axis 0
    transposed = np.moveaxis(data, 1, 0)
    img_first = Cifti2Image(transposed, header=[src_bm, series, scalar])
    out_first = parcellate(img_first, *make_parcels(), axis=0)
    assert out_first.shape == (4, 4, 3)
    assert isinstance(out_first.header.get_axis(0), ParcelsAxis)
    assert out_first.header.get_axis(1) == series
    assert out_first.header.get_axis(2) == scalar
    expected_first = reference_parcellate(transposed, src_bm, *make_parcels(), axis=0)
    np.testing.assert_allclose(np.asarray(out_first.dataobj), expected_first)
    # Same values once the parcel axis is moved back to the middle position
    np.testing.assert_allclose(
        np.moveaxis(np.asarray(out_first.dataobj), 0, 1),
        np.asarray(out_mid.dataobj),
    )


def test_chunk_sizes_agree(source_image, parcel_defs):
    img, data, src_bm, _ = source_image
    parcels, weights = parcel_defs
    reference = reference_parcellate(data, src_bm, parcels, weights)
    for chunk_size in (1, 2, 3, 7, 1000):
        out = np.asarray(parcellate(img, parcels, weights, chunk_size=chunk_size).dataobj)
        np.testing.assert_allclose(out, reference, err_msg=f'chunk_size={chunk_size}')

    # Chunking also has to work on the 3D, middle-axis layout
    series = SeriesAxis(0, 1, 4)
    scalar = ScalarAxis(['a', 'b', 'c'])
    data3d = rng.normal(size=(4, src_bm.size, 3))
    img3d = Cifti2Image(data3d, header=[series, src_bm, scalar])
    parcels3d, weights3d = parcel_defs
    reference3d = reference_parcellate(data3d, src_bm, parcels3d, weights3d, axis=1)
    for chunk_size in (1, 2, 5):
        out = np.asarray(
            parcellate(img3d, parcels3d, weights3d, axis=1, chunk_size=chunk_size).dataobj
        )
        np.testing.assert_allclose(out, reference3d)


def test_integer_source_is_not_truncated(source_image, parcel_defs):
    _, _, src_bm, series = source_image
    parcels, weights = parcel_defs
    int_data = rng.integers(-5, 6, size=(7, src_bm.size)).astype(np.int32)
    img = Cifti2Image(int_data, header=[series, src_bm])
    out = np.asarray(parcellate(img, parcels, weights).dataobj)
    assert out.dtype == np.float64
    expected = reference_parcellate(int_data, src_bm, parcels, weights)
    np.testing.assert_allclose(out, expected)
    # Fractional weighted means survive (not rounded to integers)
    assert np.any(out != np.round(out))


def test_one_dimensional_image(parcel_defs):
    src_bm = make_source_brain_model()
    data = rng.normal(size=src_bm.size)
    img = Cifti2Image(data, header=[src_bm])
    parcels, weights = parcel_defs
    out = parcellate(img, parcels, weights)
    assert out.shape == (4,)
    expected = reference_parcellate(data, src_bm, parcels, weights)
    np.testing.assert_allclose(np.asarray(out.dataobj), expected)


def test_roundtrip_preserves_parcel_axis_and_dims(tmp_path, source_image, parcel_defs):
    img, _, _, _ = source_image
    parcels, weights = parcel_defs
    result = parcellate(img, parcels, weights)
    filename = tmp_path / 'parcellated.nii'
    result.to_filename(filename)

    reloaded = nib.load(filename)
    assert reloaded.shape == (7, 4)
    assert reloaded.nifti_header.get_data_shape() == (1, 1, 1, 1, 7, 4)
    parcel_axis = reloaded.header.get_axis(1)
    assert isinstance(parcel_axis, ParcelsAxis)
    assert parcel_axis == result.header.get_axis(1)
    assert parcel_axis.name.tolist() == ['p1', 'p2', 'p3', 'p4']
    assert reloaded.header.get_axis(0) == result.header.get_axis(0)
    np.testing.assert_allclose(reloaded.get_fdata(), np.asarray(result.dataobj))


class TestValidation:
    def test_unknown_surface_member(self, source_image, parcel_defs):
        img, _, _, _ = source_image
        parcels, weights = make_parcels()
        parcels[0] = (
            'bad',
            axes.BrainModelAxis.from_surface([9], 10, 'CortexLeft'),
        )
        weights[0] = [1.0]
        with pytest.raises(ValueError, match='not part of the source brain model'):
            parcellate(img, parcels, weights)

    def test_unknown_voxel_member(self, source_image):
        img, _, _, _ = source_image
        parcels = [
            (
                'bad',
                axes.BrainModelAxis(
                    'ThalamusLeft',
                    voxel=np.array([[1, 2, 1]]),
                    affine=VOL_AFFINE,
                    volume_shape=VOL_SHAPE,
                ),
            )
        ]
        with pytest.raises(ValueError, match='not part of the source brain model'):
            parcellate(img, parcels, [[1.0]])

    def test_nvertices_conflict(self, source_image):
        img, _, _, _ = source_image
        parcel = [
            ('bad', axes.BrainModelAxis.from_surface([0], 999, 'CortexLeft')),
        ]
        with pytest.raises(ValueError, match='vertices'):
            parcellate(img, parcel, [[1.0]])

    def test_volume_grid_mismatch(self, source_image):
        img, _, _, _ = source_image
        bad_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        parcel = [
            (
                'bad',
                axes.BrainModelAxis(
                    'ThalamusLeft',
                    voxel=np.array([[0, 0, 0]]),
                    affine=bad_affine,
                    volume_shape=VOL_SHAPE,
                ),
            )
        ]
        with pytest.raises(ValueError, match='different volume grid'):
            parcellate(img, parcel, [[1.0]])

    def test_duplicated_surface_identity_within_parcel(self, source_image):
        img, _, _, _ = source_image
        parcel = [
            ('dup', axes.BrainModelAxis.from_surface([1, 1], 10, 'CortexLeft')),
        ]
        with pytest.raises(ValueError, match='duplicated surface'):
            parcellate(img, parcel, [[1.0, 1.0]])

    def test_duplicated_voxel_identity_within_parcel(self, source_image):
        img, _, _, _ = source_image
        parcel = [
            (
                'dup',
                axes.BrainModelAxis(
                    'ThalamusLeft',
                    voxel=np.array([[0, 0, 0], [0, 0, 0]]),
                    affine=VOL_AFFINE,
                    volume_shape=VOL_SHAPE,
                ),
            )
        ]
        with pytest.raises(ValueError, match='duplicated voxel'):
            parcellate(img, parcel, [[1.0, 1.0]])

    @pytest.mark.parametrize(
        ('bad_weights', 'match'),
        [
            ([[-1.0, 1.0]], 'non-negative'),
            ([[np.nan, 1.0]], 'finite'),
            ([[np.inf, 1.0]], 'finite'),
            ([[0.0, 0.0]], 'positive weight'),
        ],
    )
    def test_bad_weights(self, source_image, bad_weights, match):
        img, _, _, _ = source_image
        parcel = [
            ('p', axes.BrainModelAxis.from_surface([0, 1], 10, 'CortexLeft')),
        ]
        with pytest.raises(ValueError, match=match):
            parcellate(img, parcel, bad_weights)

    def test_weight_count_mismatch(self, source_image):
        img, _, _, _ = source_image
        parcel = [
            ('p', axes.BrainModelAxis.from_surface([0, 1], 10, 'CortexLeft')),
        ]
        with pytest.raises(ValueError, match='weights'):
            parcellate(img, parcel, [[1.0]])

    def test_empty_parcel(self, source_image):
        img, _, src_bm, _ = source_image
        empty = axes.BrainModelAxis.from_surface([], 10, 'CortexLeft')
        with pytest.raises(ValueError, match='no members'):
            parcellate(img, [('empty', empty)], [[1.0]])

    def test_bad_nan_policy_and_chunk_size(self, source_image, parcel_defs):
        img, _, _, _ = source_image
        parcels, weights = parcel_defs
        with pytest.raises(ValueError, match='nan_policy'):
            parcellate(img, parcels, weights, nan_policy='banana')
        with pytest.raises(ValueError, match='chunk_size'):
            parcellate(img, parcels, weights, chunk_size=0)

    def test_non_brain_model_axis(self):
        scalar = ScalarAxis(['a', 'b'])
        img = Cifti2Image(np.zeros((2, 2)), header=[scalar, scalar])
        parcel = [
            ('p', axes.BrainModelAxis.from_surface([0], 4, 'CortexLeft')),
        ]
        with pytest.raises(ValueError, match='BrainModelAxis'):
            parcellate(img, parcel, [[1.0]])

    def test_validation_runs_before_any_data_read(self):
        # A dataobj that explodes on access must never be reached: structural
        # validation errors take precedence and members are never dropped silently.
        src_bm = make_source_brain_model()

        class ExplodingArray:
            shape = (3, src_bm.size)
            ndim = 2
            dtype = np.float64

            def __getitem__(self, key):
                raise RuntimeError('source data should not be read')

        series = SeriesAxis(0, 1, 3)
        img = Cifti2Image(ExplodingArray(), header=[series, src_bm])
        bad_parcel = [
            ('bad', axes.BrainModelAxis.from_surface([9], 10, 'CortexLeft')),
        ]
        with pytest.raises(ValueError, match='not part of the source brain model'):
            parcellate(img, bad_parcel, [[1.0]])


class SliceTrackingArray:
    """Records every slab handed to the parcellation routine"""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.dtype = array.dtype
        self.max_slab_size = 0
        self.requested_keys = []

    def __getitem__(self, key):
        self.requested_keys.append(key)
        normalized = key if isinstance(key, tuple) else (key,)
        # Fancy indexing would defeat proxy-safe, chunked file access
        assert all(isinstance(part, slice) for part in normalized), normalized
        slab = self._array[key]
        self.max_slab_size = max(self.max_slab_size, slab.size)
        return slab


def _tracking_image(n_extra_vertices, n_parcels):
    src_bm = make_source_brain_model()
    extra = axes.BrainModelAxis.from_surface(
        np.arange(n_extra_vertices), nvertex=n_extra_vertices, name='Cerebellum'
    )
    big_bm = src_bm + extra
    n_other = 20
    data = rng.normal(size=(n_other, big_bm.size))
    tracker = SliceTrackingArray(data)
    series = SeriesAxis(0, 1, n_other)
    img = Cifti2Image(tracker, header=[series, big_bm])

    # All parcels live within the original cortex-left vertices only, so the
    # extra cerebellum greyordinates are never part of any parcel.
    parcels = []
    weights = []
    for p in range(n_parcels):
        vertices = [(p % 7), ((p + 2) % 7)]
        bm = axes.BrainModelAxis.from_surface(vertices, 10, 'CortexLeft')
        parcels.append((f'parcel-{p}', bm))
        weights.append([1.0 + (p % 3), 0.5 * (1 + p % 2)])
    return img, tracker, big_bm, parcels, weights, n_other


def test_reads_are_sparse_and_chunked_and_do_not_scale_with_parcel_count():
    # 20k unused greyordinates: a dense grey matter x parcel matrix would be ~33 MB,
    # while the sparse routine only ever touches the 8 referenced cortex-left vertices.
    n_extra = 20000
    img, tracker, _, parcels, weights, _ = _tracking_image(n_extra, 50)
    parcellate(img, parcels, weights, chunk_size=4)

    # Only cortex-left is referenced, so no read may cover the full grey matter axis
    assert tracker.requested_keys
    for key in tracker.requested_keys:
        reduction_slice = key[1]
        assert isinstance(reduction_slice, slice)
        assert reduction_slice.stop - reduction_slice.start <= 8
    # Largest slab holds at most one chunk of other positions x used grey matter
    assert tracker.max_slab_size <= 4 * 8

    # Adding many more overlapping parcels (same member union) must not enlarge reads
    img2, tracker2, _, parcels2, weights2, _ = _tracking_image(n_extra, 200)
    parcellate(img2, parcels2, weights2, chunk_size=4)
    assert tracker2.max_slab_size == tracker.max_slab_size

    # A dense grey matter x parcel implementation would grow fourfold when the parcel
    # count quadruples. Measure peak allocation for both parcel counts: the increase
    # must stay far below the extra dense-matrix cost.
    def peak_allocation(image, parcel_list, weight_list):
        tracemalloc.start()
        tracemalloc.reset_peak()
        parcellate(image, parcel_list, weight_list, chunk_size=4)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    img_small, _, _, parcels_small, weights_small, _ = _tracking_image(n_extra, 50)
    peak_small = peak_allocation(img_small, parcels_small, weights_small)
    peak_large = peak_allocation(img2, parcels2, weights2)
    dense_small = img_small.shape[1] * len(parcels_small) * 8
    dense_large = img2.shape[1] * len(parcels2) * 8
    assert dense_large - dense_small > 20 * 1024**2
    # Observed growth is only the small, linear per-parcel bookkeeping overhead
    assert peak_large - peak_small < (dense_large - dense_small) * 0.05
    # And the large run itself never approaches the dense matrix footprint
    assert peak_large < dense_large * 0.5
