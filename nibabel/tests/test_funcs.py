# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test for image funcs"""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..analyze import AnalyzeImage
from ..arrayproxy import is_proxy
from ..funcs import CatImgError, CatProxy, OrientationError, as_closest_canonical, concat_images
from ..loadsave import load, save
from ..nifti1 import Nifti1Image
from ..tmpdirs import InTemporaryDirectory

_counter = 0


def _as_fname(img):
    global _counter
    fname = f'img{_counter:3d}.nii'
    _counter = _counter + 1
    save(img, fname)
    return fname


class _CountingProxy:
    """Wrap a dataobj and count calls that read voxel data"""

    is_proxy = True

    def __init__(self, prox):
        self._prox = prox
        self.n_reads = 0

    @property
    def shape(self):
        return self._prox.shape

    @property
    def ndim(self):
        return self._prox.ndim

    @property
    def dtype(self):
        return self._prox.dtype

    @property
    def slope(self):
        return self._prox.slope

    @property
    def inter(self):
        return self._prox.inter

    def __getitem__(self, slicer):
        self.n_reads += 1
        return self._prox[slicer]

    def __array__(self, dtype=None):
        self.n_reads += 1
        arr = np.asanyarray(self._prox)
        return arr if dtype is None else arr.astype(dtype)


@pytest.mark.thread_unsafe
def test_concat():
    # Smoke test: concat empty list.
    with pytest.raises(ValueError):
        concat_images([])

    # Build combinations of 3D, 4D w/size[3] == 1, and 4D w/size[3] == 3
    all_shapes_5D = ((1, 4, 5, 3, 3), (7, 3, 1, 4, 5), (0, 2, 1, 4, 5))

    affine = np.eye(4)
    for dim in range(2, 6):
        all_shapes_ND = tuple(shape[:dim] for shape in all_shapes_5D)
        all_shapes_N1D_unary = tuple(shape + (1,) for shape in all_shapes_ND)
        all_shapes = all_shapes_ND + all_shapes_N1D_unary

        # Loop over all possible combinations of images, in first and
        #   second position.
        for data0_shape in all_shapes:
            data0_numel = np.asarray(data0_shape).prod()
            data0 = np.arange(data0_numel, dtype='int32').reshape(data0_shape)
            img0_mem = Nifti1Image(data0, affine)

            for data1_shape in all_shapes:
                data1_numel = np.asarray(data1_shape).prod()
                data1 = np.arange(data1_numel, dtype='int32').reshape(data1_shape)
                img1_mem = Nifti1Image(data1, affine)
                img2_mem = Nifti1Image(data1, affine + 1)  # bad affine

                # Loop over every possible axis, including None (explicit and implied)
                for axis in list(range(-(dim - 2), (dim - 1))) + [None, '__default__']:
                    # Allow testing default vs. passing explicit param
                    if axis == '__default__':
                        np_concat_kwargs = dict(axis=-1)
                        concat_imgs_kwargs = dict()
                        axis = None  # Convert downstream
                    elif axis is None:
                        np_concat_kwargs = dict(axis=-1)
                        concat_imgs_kwargs = dict(axis=axis)
                    else:
                        np_concat_kwargs = dict(axis=axis)
                        concat_imgs_kwargs = dict(axis=axis)

                    # Create expected output
                    try:
                        # Error will be thrown if the np.concatenate fails.
                        #   However, when axis=None, the concatenate is possible
                        #   but our efficient logic (where all images are
                        #   3D and the same size) fails, so we also
                        #   have to expect errors for those.
                        if axis is None:  # 3D from here and below
                            all_data = np.concatenate(
                                [data0[..., np.newaxis], data1[..., np.newaxis]],
                                **np_concat_kwargs,
                            )
                        else:  # both 3D, appending on final axis
                            all_data = np.concatenate([data0, data1], **np_concat_kwargs)
                        expect_error = False
                    except ValueError:
                        # Shapes are not combinable
                        expect_error = True

                    # Check filenames and in-memory images work
                    with InTemporaryDirectory():
                        # Try mem-based, file-based, and mixed
                        imgs = [img0_mem, img1_mem, img2_mem]
                        img_files = [_as_fname(img) for img in imgs]
                        imgs_mixed = [imgs[0], img_files[1], imgs[2]]
                        for img0, img1, img2 in (imgs, img_files, imgs_mixed):
                            try:
                                all_imgs = concat_images([img0, img1], **concat_imgs_kwargs)
                            except ValueError as ve:
                                assert expect_error, str(ve)
                            else:
                                assert not expect_error, (
                                    'Expected a concatenation error, but got none.'
                                )
                                assert_array_equal(all_imgs.get_fdata(), all_data)
                                assert_array_equal(all_imgs.affine, affine)

                            # check that not-matching affines raise error
                            with pytest.raises(ValueError):
                                concat_images([img0, img2], **concat_imgs_kwargs)

                            # except if check_affines is False
                            try:
                                all_imgs = concat_images([img0, img1], **concat_imgs_kwargs)
                            except ValueError as ve:
                                assert expect_error, str(ve)
                            else:
                                assert not expect_error, (
                                    'Expected a concatenation error, but got none.'
                                )
                                assert_array_equal(all_imgs.get_fdata(), all_data)
                                assert_array_equal(all_imgs.affine, affine)


def _shift_aff(affine, axis, offset):
    shift = np.eye(4)
    shift[axis, 3] = offset
    return affine.dot(shift)


def test_concat_lazy_matches_model():
    # Lazy concat matches the numpy model and the eager implementation, for
    # in-memory images, file-backed proxies and mixed inputs
    for dim in range(2, 5):
        shape = (2, 3, 4, 2, 1)[:dim]
        d0 = np.arange(np.prod(shape), dtype='int32').reshape(shape)
        d1 = (d0 + np.prod(shape)).astype('int32')
        with InTemporaryDirectory():
            f0, f1 = _as_fname(Nifti1Image(d0, np.eye(4))), _as_fname(Nifti1Image(d1, np.eye(4)))
            proxy0, proxy1 = load(f0, mmap=False).dataobj, load(f1, mmap=False).dataobj
            hdr0, hdr1 = load(f0, mmap=False).header, load(f1, mmap=False).header
            for axis in list(range(-dim, dim)) + [None]:
                norm_axis = dim - 1 if axis is None else (axis + dim if axis < 0 else axis)
                spatial = axis is not None and norm_axis < 3
                if spatial:
                    # Spatial concatenation: build a continuous second image
                    aff0 = np.eye(4)
                    aff1 = _shift_aff(aff0, norm_axis, shape[norm_axis])
                    lazy_kw = {'allow_spatial': True}
                else:
                    aff0 = aff1 = np.eye(4)
                    lazy_kw = {}
                img0 = Nifti1Image(d0, aff0)
                img1 = Nifti1Image(d1, aff1)
                p0 = Nifti1Image(proxy0, aff0, hdr0.copy())
                p1 = Nifti1Image(proxy1, aff1, hdr1.copy())
                if axis is None:
                    expected = np.concatenate([d0[..., None], d1[..., None]], axis=-1)
                else:
                    expected = np.concatenate([d0, d1], axis=axis)
                for a, b in ((img0, img1), (p0, p1), (img0, p1)):
                    result = concat_images([a, b], axis=axis, lazy=True, **lazy_kw)
                    assert is_proxy(result.dataobj)
                    assert isinstance(result.dataobj, CatProxy)
                    assert result.shape == expected.shape
                    assert result.header.get_data_shape() == expected.shape
                    assert_array_equal(np.asanyarray(result.dataobj), expected)
                    # Compare against eager concat; the spatial case uses
                    # affines that eager only accepts with its check disabled
                    eager = concat_images([a, b], axis=axis, check_affines=not spatial)
                    assert_array_equal(result.get_fdata(), eager.get_fdata())


def test_concat_lazy_construction_reads_no_voxels():
    # Only shape / affine / header metadata are inspected at construction
    affine = np.eye(4)
    d0 = np.arange(2 * 3 * 4 * 2, dtype='int16').reshape(2, 3, 4, 2)
    d1 = np.arange(2 * 3 * 4 * 3, dtype='int16').reshape(2, 3, 4, 3)
    with InTemporaryDirectory():
        f0, f1 = _as_fname(Nifti1Image(d0, affine)), _as_fname(Nifti1Image(d1, affine))
        src0, src1 = load(f0, mmap=False), load(f1, mmap=False)
        c0, c1 = _CountingProxy(src0.dataobj), _CountingProxy(src1.dataobj)
        w0 = Nifti1Image(c0, src0.affine, src0.header)
        w1 = Nifti1Image(c1, src1.affine, src1.header)
        result = concat_images([w0, w1], axis=3, lazy=True)
        assert c0.n_reads == 0 and c1.n_reads == 0
        assert not src0.in_memory and not src1.in_memory
        # Metadata snapshot, not voxel data
        assert result.shape == (2, 3, 4, 5)
        # Slicing still leaves source caches untouched
        result.dataobj[..., 0]
        result.get_fdata(caching='unchanged')
        assert not src0.in_memory and not src1.in_memory
        # The source files remain openable / readable through the source images
        assert_array_equal(np.asanyarray(src0.dataobj), d0)


def test_concat_lazy_only_hits_selected_proxies():
    affine = np.eye(4)
    d0 = np.arange(2 * 2 * 2 * 2, dtype='int16').reshape(2, 2, 2, 2)
    d1 = d0 + 100
    d2 = d0 + 200
    with InTemporaryDirectory():
        files = [_as_fname(Nifti1Image(d, affine)) for d in (d0, d1, d2)]
        counters = []
        imgs = []
        for fname in files:
            src = load(fname, mmap=False)
            counter = _CountingProxy(src.dataobj)
            counters.append(counter)
            imgs.append(Nifti1Image(counter, src.affine, src.header))
        cat = concat_images(imgs, axis=3, lazy=True)  # offsets 0, 2, 4; total 6
        # Slice entirely inside the second child
        cat.dataobj[..., 2:4]
        assert [c.n_reads for c in counters] == [0, 1, 0]
        # Slice spanning the first and third child but not the second
        for c in counters:
            c.n_reads = 0
        cat.dataobj[(0, 0, 0, slice(0, 6, 4))]  # global indices 0 and 4
        assert [c.n_reads for c in counters] == [1, 0, 1]
        # Empty selection performs no I/O at all
        for c in counters:
            c.n_reads = 0
        empty = cat.dataobj[..., 6:10]
        assert empty.size == 0
        assert empty.shape == (2, 2, 2, 0)
        assert [c.n_reads for c in counters] == [0, 0, 0]
        # Full reverse visits every child exactly once, in reverse order
        for c in counters:
            c.n_reads = 0
        full = np.concatenate([d0, d1, d2], axis=3)
        assert_array_equal(cat.dataobj[..., ::-1], full[..., ::-1])
        assert [c.n_reads for c in counters] == [1, 1, 1]


def test_concat_lazy_scaling_and_dtype():
    # Each child's slope / inter is applied once; output dtype represents all
    # children and is not narrowed when only one child happens to be selected
    affine = np.eye(4)
    d0 = np.arange(1, 9, dtype='int16').reshape(2, 2, 2, 1)
    d1 = np.arange(1, 9, dtype='int32').reshape(2, 2, 2, 1) * 1000
    img0 = Nifti1Image(d0, affine)
    img0.header.set_slope_inter(2.0, 1.0)
    img1 = Nifti1Image(d1, affine)
    with InTemporaryDirectory():
        f0, f1 = _as_fname(img0), _as_fname(img1)
        src0, src1 = load(f0, mmap=False), load(f1, mmap=False)
        cat = concat_images([src0, src1], axis=3, lazy=True)
        # Header must not re-apply the first image's scaling
        assert cat.header.get_slope_inter() == (None, None)
        scaled0 = np.asanyarray(src0.dataobj)
        scaled1 = np.asanyarray(src1.dataobj)
        expected = np.concatenate([scaled0, scaled1], axis=3)
        got = np.asanyarray(cat.dataobj)
        assert_array_equal(got, expected)
        assert got.dtype == np.promote_types(scaled0.dtype, scaled1.dtype)
        # A slice hitting only the second child keeps the combined dtype
        sub = cat.dataobj[..., 1]
        assert sub.dtype == got.dtype
        assert_array_equal(sub, scaled1[..., 0])
        # Explicit __array__ materialization honors the requested dtype
        assert np.asarray(cat.dataobj, dtype=np.float32).dtype == np.float32


def test_concat_lazy_indexing():
    affine = np.eye(4)
    d0 = np.arange(2 * 3 * 3 * 2, dtype='int16').reshape(2, 3, 3, 2)
    d1 = d0 + 1000
    d2 = d0 + 2000
    imgs = [Nifti1Image(d0, affine), Nifti1Image(d1, affine), Nifti1Image(d2, affine)]
    cat = concat_images(imgs, axis=3, lazy=True)  # total length 6
    full = np.concatenate([d0, d1, d2], axis=3)
    # Boundary integer indices, including the exact child offsets
    for index in (0, 1, 2, 3, 4, 5, -1, -2, -6):
        assert_array_equal(cat.dataobj[..., index], full[..., index])
    # Negative steps that cross child boundaries
    for slicer in (
        slice(None, None, -1),
        slice(None, None, -2),
        slice(5, 0, -1),
        slice(5, 1, -2),
        slice(4, 1, -1),
        slice(1, 5, 2),
    ):
        assert_array_equal(cat.dataobj[..., slicer], full[..., slicer])
    # Ellipsis and new axes
    for slicer in (..., (..., None), (None, ...), (0, ..., None), (Ellipsis, 0, None)):
        assert_array_equal(cat.dataobj[slicer], full[slicer])
        assert cat.dataobj[slicer].shape == full[slicer].shape
    # Fancy indexing is rejected, as for ArrayProxy
    with pytest.raises(ValueError):
        cat.dataobj[..., np.array([0, 2])]


def test_concat_lazy_stack_new_axis():
    # axis=None stacks on a new trailing axis without touching voxels
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    d0 = np.arange(2 * 3 * 4, dtype='int16').reshape(2, 3, 4)
    d1 = d0 + 100
    with InTemporaryDirectory():
        f0, f1 = _as_fname(Nifti1Image(d0, affine)), _as_fname(Nifti1Image(d1, affine))
        src0, src1 = load(f0, mmap=False), load(f1, mmap=False)
        c0, c1 = _CountingProxy(src0.dataobj), _CountingProxy(src1.dataobj)
        w0 = Nifti1Image(c0, src0.affine, src0.header)
        w1 = Nifti1Image(c1, src1.affine, src1.header)
        cat = concat_images([w0, w1], lazy=True)
        assert c0.n_reads == 0 and c1.n_reads == 0
        # New shape comes from the proxy, not the first header's old shape
        assert cat.shape == (2, 3, 4, 2)
        assert cat.header.get_data_shape() == (2, 3, 4, 2)
        # Spatial affine is untouched; new axis gets a unit zoom
        assert_array_equal(cat.affine, affine)
        assert cat.header.get_zooms() == (2.0, 3.0, 4.0, 1.0)
        expected = np.concatenate([d0[..., None], d1[..., None]], axis=-1)
        assert_array_equal(np.asanyarray(cat.dataobj), expected)
        # Reverse and integer indexing on the stack axis map to the right child
        assert_array_equal(cat.dataobj[..., ::-1], expected[..., ::-1])
        assert_array_equal(cat.dataobj[..., 1], d1)
        # Full read + reverse touched both children; the integer index only
        # ever reaches the second child
        assert [c0.n_reads, c1.n_reads] == [2, 3]


def test_concat_lazy_time_checks():
    affine = np.eye(4)
    data = np.ones((2, 2, 2, 3), dtype='int16')
    img0 = Nifti1Image(data, affine)
    # Mismatched time step
    img1 = Nifti1Image(data, affine)
    img1.header.set_zooms((1.0, 1.0, 1.0, 2.5))
    with pytest.raises(CatImgError):
        concat_images([img0, img1], axis=3, lazy=True)
    # Mismatched time units
    img1 = Nifti1Image(data, affine)
    img0.header.set_xyzt_units(t='sec')
    img1.header.set_xyzt_units(t='msec')
    with pytest.raises(CatImgError):
        concat_images([img0, img1], axis=3, lazy=True)
    # Mismatched spatial units
    img1 = Nifti1Image(data, affine)
    img0.header.set_xyzt_units(xyz='mm', t='sec')
    img1.header.set_xyzt_units(xyz='meter', t='sec')
    with pytest.raises(CatImgError):
        concat_images([img0, img1], axis=3, lazy=True)
    # Matching time metadata is accepted and keeps the time zoom
    img1 = Nifti1Image(data, affine)
    img0.header.set_xyzt_units(xyz='mm', t='sec')
    img1.header.set_xyzt_units(xyz='mm', t='sec')
    cat = concat_images([img0, img1], axis=3, lazy=True)
    assert cat.header.get_zooms()[3] == img0.header.get_zooms()[3]


def test_concat_lazy_spatial_rules():
    affine = np.eye(4)
    d0 = np.ones((2, 2, 2), dtype='int16')
    d1 = np.ones((2, 2, 2), dtype='int16') * 9
    img0 = Nifti1Image(d0, affine)
    img1 = Nifti1Image(d1, affine)
    # Spatial concatenation requires explicit permission
    with pytest.raises(CatImgError):
        concat_images([img0, img1], axis=0, lazy=True)
    # Identical affines do not make the blocks voxel-to-world continuous
    with pytest.raises(CatImgError):
        concat_images([img0, img1], axis=0, lazy=True, allow_spatial=True)
    # A correctly shifted affine is continuous: block 1 starts at voxel 2
    img1 = Nifti1Image(d1, _shift_aff(affine, 0, 2))
    cat = concat_images([img0, img1], axis=0, lazy=True, allow_spatial=True)
    assert cat.shape == (4, 2, 2)
    assert_array_equal(cat.affine, affine)
    assert_array_equal(cat.get_fdata(), np.concatenate([d0, d1], axis=0))
    # A wrongly shifted affine is still rejected
    img_bad = Nifti1Image(d1, _shift_aff(affine, 0, 3))
    with pytest.raises(CatImgError):
        concat_images([img0, img_bad], axis=0, lazy=True, allow_spatial=True)


def test_concat_lazy_validation():
    affine = np.eye(4)
    img0 = Nifti1Image(np.ones((2, 2, 2), dtype='int16'), affine)
    img1 = Nifti1Image(np.ones((2, 2, 3), dtype='int16'), affine)
    # Dimensionality mismatch
    with pytest.raises(ValueError):
        concat_images([img0, Nifti1Image(np.ones((2, 2)), affine)], axis=0, lazy=True)
    # Non-concatenation axis size mismatch
    with pytest.raises(ValueError):
        concat_images([img0, img1], axis=2, lazy=True, allow_spatial=True)
    # Out-of-range axis
    with pytest.raises(CatImgError):
        concat_images([img0, img0], axis=5, lazy=True)
    # Affine mismatch on a non-spatial axis
    img_bad = Nifti1Image(np.ones((2, 2, 2, 1), dtype='int16'), affine + 1)
    img_ref = Nifti1Image(np.ones((2, 2, 2, 1), dtype='int16'), affine)
    with pytest.raises(ValueError):
        concat_images([img_ref, img_bad], axis=3, lazy=True)
    # ...but check_affines=False restores the permissive behavior
    concat_images([img_ref, img_bad], axis=3, lazy=True, check_affines=False)


def test_concat_lazy_pickle_reopens_files():
    affine = np.eye(4)
    d0 = np.arange(2 * 2 * 2 * 2, dtype='int16').reshape(2, 2, 2, 2)
    d1 = d0 + 100
    with InTemporaryDirectory():
        f0, f1 = _as_fname(Nifti1Image(d0, affine)), _as_fname(Nifti1Image(d1, affine))
        cat = concat_images([load(f0, mmap=False), load(f1, mmap=False)], axis=3, lazy=True)
        expected = np.asanyarray(cat.dataobj)
        restored = pickle.loads(pickle.dumps(cat.dataobj))
        assert isinstance(restored, CatProxy)
        assert restored.shape == cat.shape
        # Children recover via their own file strategies and re-read correctly
        assert_array_equal(np.asanyarray(restored), expected)


def test_closest_canonical():
    # Use 32-bit data so that the AnalyzeImage class doesn't complain
    arr = np.arange(24, dtype=np.int32).reshape((2, 3, 4, 1))

    # Test with an AnalyzeImage first
    img = AnalyzeImage(arr, np.eye(4))
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img

    # And a case where the Analyze image has to be flipped
    img = AnalyzeImage(arr, np.diag([-1, 1, 1, 1]))
    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.flipud(arr))

    # Now onto the NIFTI cases (where dim_info also has to be updated)

    # No funky stuff, returns same thing
    img = Nifti1Image(arr, np.eye(4))
    # set freq/phase/slice dim so that we can check that we
    # re-order them properly
    img.header.set_dim_info(0, 1, 2)
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img

    # a axis flip
    img = Nifti1Image(arr, np.diag([-1, 1, 1, 1]))
    img.header.set_dim_info(0, 1, 2)
    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    assert img.header.get_dim_info() == xyz_img.header.get_dim_info()
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.flipud(arr))

    # no error for enforce_diag in this case
    xyz_img = as_closest_canonical(img, True)
    # but there is if the affine is not diagonal
    aff = np.eye(4)
    aff[0, 1] = 0.1
    # although it's more or less canonical already
    img = Nifti1Image(arr, aff)
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img
    # it's still not diagonal
    with pytest.raises(OrientationError):
        as_closest_canonical(img, True)

    # an axis swap
    aff = np.diag([1, 0, 0, 1])
    aff[1, 2] = 1
    aff[2, 1] = 1
    img = Nifti1Image(arr, aff)
    img.header.set_dim_info(0, 1, 2)

    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    # Check both the original and new objects
    assert img.header.get_dim_info() == (0, 1, 2)
    assert xyz_img.header.get_dim_info() == (0, 2, 1)
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.transpose(arr, (0, 2, 1, 3)))

    # same axis swap but with None dim info (except for slice dim)
    img.header.set_dim_info(None, None, 2)
    xyz_img = as_closest_canonical(img)
    assert xyz_img.header.get_dim_info() == (None, None, 1)
