# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Processor functions for images"""

from bisect import bisect_right
from numbers import Integral

import numpy as np

from .arrayproxy import get_obj_dtype
from .fileslice import canonical_slicers, fill_slicer, predict_shape
from .loadsave import load
from .orientations import OrientationError, io_orientation
from .volumeutils import int_scinter_ftype


class CatImgError(ValueError):
    """Error while building a lazy concatenation of images"""


def _ceil_div(a, b):
    """Ceiling of ``a / b`` for positive ``b``, without float conversion"""
    return -((-a) // b)


def _scaled_dtype(obj):
    """Dtype of the data returned by `obj`, after its scaling is applied

    Unlike :func:`nibabel.arrayproxy.get_obj_dtype`, this routine never reads
    voxel data; it uses the scaling metadata exposed by array proxies.  Returns
    ``None`` if the output dtype cannot be determined without reading data.
    """
    # ArrayProxy instances expose their storage dtype and scaling factors
    slope = getattr(obj, 'slope', None)
    inter = getattr(obj, 'inter', None)
    if slope is not None:
        in_dtype = np.dtype(obj.dtype)
        if (slope, inter) == (1.0, 0.0):
            return in_dtype
        slope_arr = np.asanyarray(slope)
        inter_arr = np.asanyarray(inter)
        if in_dtype.kind in 'iu':
            # Mirror ArrayProxy._get_scaled / apply_read_scaling
            default = slope_arr.dtype.type if slope_arr.dtype.kind == 'f' else np.float64
            return np.dtype(int_scinter_ftype(in_dtype, slope_arr, inter_arr, default))
        return np.result_type(in_dtype, slope_arr.dtype, inter_arr.dtype)
    if hasattr(obj, 'dtype'):
        return np.dtype(obj.dtype)
    return None


class CatProxy:
    """Lazy proxy combining the dataobjs of several images

    The proxy maps global indices on the concatenation axis back to the child
    dataobj actually holding each voxel.  Only children hit by a given slice
    are read; each child's slope / intercept scaling is applied once, on its
    own read path.

    Parameters
    ----------
    dataobjs : sequence of array-like
        Child data proxies / arrays, in caller (concatenation) order.
    axis : int
        Axis of the combined array along which the children are joined.  An
        `axis` equal to the child dimensionality selects stacking on a new
        trailing axis.
    shapes : sequence of tuple
        Snapshot of each child shape.
    offsets : sequence of int
        Start index of each child on `axis`, plus the total length as the
        final element.
    """

    is_proxy = True

    def __init__(self, dataobjs, axis, shapes, offsets):
        self._dataobjs = tuple(dataobjs)
        self._axis = int(axis)
        self._shapes = tuple(tuple(shape) for shape in shapes)
        self._offsets = tuple(int(offset) for offset in offsets)
        out_shape = list(self._shapes[0])
        if self._axis == len(out_shape):  # New trailing stack axis
            out_shape.append(len(self._dataobjs))
        else:
            out_shape[self._axis] = self._offsets[-1]
        self._shape = tuple(out_shape)
        self._dtype = self._infer_dtype()

    def _infer_dtype(self):
        """Promoted output dtype from all children, without reading voxels"""
        dtypes = []
        for obj in self._dataobjs:
            dtype = _scaled_dtype(obj)
            if dtype is None:
                # A proxy without scaling metadata; defer until first read
                return None
            dtypes.append(dtype)
        dtype = dtypes[0]
        for next_dtype in dtypes[1:]:
            dtype = np.promote_types(dtype, next_dtype)
        return np.dtype(dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        if self._dtype is None:
            # Only reached for proxies that expose no scaling metadata
            dtypes = [get_obj_dtype(obj) for obj in self._dataobjs]
            dtype = dtypes[0]
            for next_dtype in dtypes[1:]:
                dtype = np.promote_types(dtype, next_dtype)
            self._dtype = np.dtype(dtype)
        return self._dtype

    def __array__(self, dtype=None):
        # Explicit materialization of the full combined result
        arr = self[()].astype(self.dtype, copy=False)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, slicer):
        # Canonicalize: expand Ellipsis / missing axes, make ints positive,
        # keep None (new axis) markers; rejects fancy indexing as ArrayProxy
        canonical = canonical_slicers(slicer, self._shape)
        real_slicers = tuple(s for s in canonical if s is not None)
        axis_slicer = real_slicers[self._axis]
        if isinstance(axis_slicer, Integral):
            hits = self._hits_for_int(int(axis_slicer))
        else:
            hits = self._hits_for_slice(fill_slicer(axis_slicer, self._shape[self._axis]))
        if not hits:
            # Empty selection: no child proxy is touched
            return np.empty(predict_shape(canonical, self._shape), dtype=self.dtype)
        # Axis of the joined / stacked dimension within the child arrays
        # (integer indices on earlier axes drop dimensions, new axes are
        # absent in children)
        child_axis = sum(not isinstance(real_slicers[ax], Integral) for ax in range(self._axis))
        stack = self._axis == self._dataobjs[0].ndim
        pieces = []
        for child_idx, local_axis_slicer in hits:
            if stack:
                # Children do not have the stack axis; it only selects children
                local_slicers = real_slicers[: self._axis]
            else:
                local_slicers = list(real_slicers)
                local_slicers[self._axis] = local_axis_slicer
            pieces.append(self._dataobjs[child_idx][tuple(local_slicers)])
        if isinstance(axis_slicer, Integral):
            arr = pieces[0]
        elif stack:
            arr = np.stack(pieces, axis=child_axis)
        else:
            arr = np.concatenate(pieces, axis=child_axis)
        # Keep the dtype stable, independent of which children were hit
        arr = arr.astype(self.dtype, copy=False)
        # Re-insert any new axes requested via None entries; integer-indexed
        # real axes have already been dropped from `arr`
        if any(s is None for s in canonical):
            new_axes = tuple(
                None if entry is None else slice(None)
                for entry in canonical
                if entry is None or not isinstance(entry, Integral)
            )
            arr = arr[new_axes]
        return arr

    def _hits_for_int(self, index):
        child = bisect_right(self._offsets, index) - 1
        return [(child, index - self._offsets[child])]

    def _hits_for_slice(self, full):
        """Return ``(child index, local slicer)`` pairs in global visit order

        ``full`` is a filled slice (see :func:`fill_slicer`) on the combined
        axis.  Pairs are ordered as the global index sequence visits them, so
        negative steps visit children in reverse order.
        """
        start, stop, step = full.start, full.stop, full.step
        n_children = len(self._dataobjs)
        hits = []
        if step > 0:
            for child in range(n_children):
                lo = self._offsets[child]
                hi = self._offsets[child + 1]
                end = min(stop, hi)
                if start >= end:
                    continue
                n_steps = max(0, _ceil_div(lo - start, step))
                first = start + step * n_steps
                if first >= end:
                    continue
                hits.append((child, slice(first - lo, end - lo, step)))
        else:
            neg_step = -step
            stop_value = -1 if stop is None else stop
            for child in range(n_children - 1, -1, -1):
                lo = self._offsets[child]
                hi = self._offsets[child + 1]
                n_steps = max(0, _ceil_div(start - (hi - 1), neg_step))
                first = start - n_steps * neg_step
                lower = max(stop_value, lo - 1)
                if first <= lower:
                    continue
                # Use an explicit local stop; a negative stop would be
                # reinterpreted by the child relative to its own axis length
                local_stop = None if lower == lo - 1 else lower - lo
                hits.append((child, slice(first - lo, local_stop, step)))
        return hits


def squeeze_image(img):
    """Return image, remove axes length 1 at end of image shape

    For example, an image may have shape (10,20,30,1,1).  In this case
    squeeze will result in an image with shape (10,20,30).  See doctests
    for further description of behavior.

    Parameters
    ----------
    img : ``SpatialImage``

    Returns
    -------
    squeezed_img : ``SpatialImage``
       Copy of img, such that data, and data shape have been squeezed,
       for dimensions > 3rd, and at the end of the shape list

    Examples
    --------
    >>> import nibabel as nf
    >>> shape = (10,20,30,1,1)
    >>> data = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    >>> affine = np.eye(4)
    >>> img = nf.Nifti1Image(data, affine)
    >>> img.shape == (10, 20, 30, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 20, 30)
    True

    If the data are 3D then last dimensions of 1 are ignored

    >>> shape = (10,1,1)
    >>> data = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (10, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 1, 1)
    True

    Only *final* dimensions of 1 are squeezed

    >>> shape = (1, 1, 5, 1, 2, 1, 1)
    >>> data = data.reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (1, 1, 5, 1, 2, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (1, 1, 5, 1, 2)
    True
    """
    klass = img.__class__
    shape = img.shape
    slen = len(shape)
    if slen < 4:
        return klass.from_image(img)
    for bdim in shape[3::][::-1]:
        if bdim == 1:
            slen -= 1
        else:
            break
    if slen == len(shape):
        return klass.from_image(img)
    shape = shape[:slen]
    data = np.asanyarray(img.dataobj).reshape(shape)
    return klass(data, img.affine, img.header, img.extra)


def _header_units(header):
    """Spatial / temporal unit labels of `header`, or ``None`` if unavailable"""
    if hasattr(header, 'get_xyzt_units'):
        return header.get_xyzt_units()
    return None


def _shift_affine(affine, axis, offset):
    """Affine mapping voxels shifted by `offset` indices along `axis`"""
    shift = np.eye(4)
    shift[axis, 3] = offset
    return np.asarray(affine).dot(shift)


def _concat_images_lazy(images, axis, check_affines, allow_spatial):
    """Build a proxy-backed concatenation without reading image voxels"""
    img0 = images[0]
    klass = img0.__class__
    shape0 = tuple(img0.shape)
    n_dim = len(shape0)
    if axis is None:
        stack = True
        cat_axis = n_dim  # New trailing axis
    else:
        stack = False
        cat_axis = axis + n_dim if axis < 0 else axis
        if not 0 <= cat_axis < n_dim:
            raise CatImgError(f'axis {axis} is out of bounds for {n_dim}-dimensional images')
    spatial_concat = not stack and cat_axis < 3

    aff0 = np.array(img0.affine, copy=True)
    hdr0 = img0.header
    zooms0 = tuple(hdr0.get_zooms())
    units0 = _header_units(hdr0)
    n_spatial = min(3, n_dim)

    shapes = []
    offsets = [0]
    dataobjs = []
    for i, img in enumerate(images):
        shape = tuple(img.shape)
        if len(shape) != n_dim:
            raise ValueError(f'Image {i} has {len(shape)} dimensions, image 0 has {n_dim}')
        if stack:
            if shape != shape0:
                raise ValueError(
                    f'shape {shape} for image {i} not compatible with '
                    f'first image shape {shape0} with axis == None'
                )
        else:
            for ax in range(n_dim):
                if ax != cat_axis and shape[ax] != shape0[ax]:
                    raise ValueError(
                        f'shape {shape} for image {i} not compatible with '
                        f'first image shape {shape0} with axis == {axis}'
                    )

        header = img.header
        units = _header_units(header)
        if units is not None and units != units0:
            raise CatImgError(f'Image {i} units {units} do not match image 0 units {units0}')
        zooms = tuple(header.get_zooms())
        if check_affines and zooms[:n_spatial] != zooms0[:n_spatial]:
            raise CatImgError(
                f'Image {i} spatial voxel sizes {zooms[:n_spatial]} do not match '
                f'image 0 spatial voxel sizes {zooms0[:n_spatial]}'
            )
        if not stack and cat_axis >= 3 and len(zooms) > cat_axis:
            if zooms[cat_axis] != zooms0[cat_axis]:
                raise CatImgError(
                    f'Image {i} step size {zooms[cat_axis]} on axis {cat_axis} does '
                    f'not match image 0 step size {zooms0[cat_axis]}'
                )

        if spatial_concat:
            if not allow_spatial:
                raise CatImgError(
                    f'Concatenation on spatial axis {cat_axis} is only accepted with '
                    'allow_spatial=True, and requires voxel-to-world continuity'
                )
            expected_aff = _shift_affine(aff0, cat_axis, offsets[i])
            if not np.allclose(np.asarray(img.affine), expected_aff):
                raise CatImgError(
                    f'Affine for image {i} is not continuous with the preceding '
                    f'images along spatial axis {cat_axis}'
                )
        elif check_affines and not np.allclose(np.asarray(img.affine), aff0):
            raise ValueError(f'Affine for image {i} does not match affine for first image')

        shapes.append(shape)
        offsets.append(offsets[-1] + (1 if stack else shape[cat_axis]))
        # Keep only the dataobj; source image caches / file handles are left alone
        dataobjs.append(img.dataobj)

    proxy = CatProxy(dataobjs, cat_axis, shapes, offsets)

    # Snapshot the first header, then make it describe the combined data rather
    # than any single source image
    out_header = klass.header_class.from_header(hdr0)
    if hasattr(out_header, 'set_slope_inter'):
        # Scaling is applied per child inside the proxy; do not re-apply the
        # first image's slope / intercept on read
        out_header.set_slope_inter(None)
    # Sets the combined shape (and a unit zoom for a new stack axis)
    out_header.set_data_shape(proxy.shape)

    return klass(proxy, aff0, out_header)


def concat_images(images, check_affines=True, axis=None, lazy=False, allow_spatial=False):
    r"""Concatenate images in list to single image, along specified dimension

    Parameters
    ----------
    images : sequence
       sequence of ``SpatialImage`` or filenames of the same dimensionality\s
    check_affines : {True, False}, optional
       If True, then check that all the affines for `images` are nearly
       the same, raising a ``ValueError`` otherwise.  Default is True
    axis : None or int, optional
        If None, concatenates on a new dimension.  This requires all images to
        be the same shape.  If not None, concatenates on the specified
        dimension.  This requires all images to be the same shape, except on
        the specified dimension.
    lazy : {False, True}, optional
        If True, build the result from the ``dataobj`` proxies without reading
        any image voxel data.  Only shape, affine and the header metadata
        needed for validation are inspected.  Voxels are read on demand from
        the returned proxy, and only from the child images hit by each slice.
        The default (False) keeps the original eager behavior.
    allow_spatial : {False, True}, optional
        Only used when `lazy` is True.  Allow concatenation along one of the
        first three (spatial) axes.  Acceptance additionally requires that the
        images are voxel-to-world continuous across the concatenation
        boundary; two blocks sharing an identical affine are therefore
        rejected as spatially adjacent blocks.

    Returns
    -------
    concat_img : ``SpatialImage``
       New image resulting from concatenating `images` across last
       dimension
    """
    images = [load(img) if not hasattr(img, 'get_data') else img for img in images]
    n_imgs = len(images)
    if n_imgs == 0:
        raise ValueError('Cannot concatenate an empty list of images.')
    if lazy:
        return _concat_images_lazy(images, axis, check_affines, allow_spatial)
    img0 = images[0]
    affine = img0.affine
    header = img0.header
    klass = img0.__class__
    shape0 = img0.shape
    n_dim = len(shape0)
    if axis is None:
        # collect images in output array for efficiency
        out_shape = (n_imgs,) + shape0
        out_data = np.empty(out_shape)
    else:
        # collect images in list for use with np.concatenate
        out_data = [None] * n_imgs
    # Get part of shape we need to check inside loop
    idx_mask = np.ones((n_dim,), dtype=bool)
    if axis is not None:
        idx_mask[axis] = False
    masked_shape = np.array(shape0)[idx_mask]
    for i, img in enumerate(images):
        if len(img.shape) != n_dim:
            raise ValueError(f'Image {i} has {len(img.shape)} dimensions, image 0 has {n_dim}')
        if not np.all(np.array(img.shape)[idx_mask] == masked_shape):
            raise ValueError(
                f'shape {img.shape} for image {i} not compatible with '
                f'first image shape {shape0} with axis == {axis}'
            )
        if check_affines and not np.all(img.affine == affine):
            raise ValueError(f'Affine for image {i} does not match affine for first image')
        # Do not fill cache in image if it is empty
        out_data[i] = np.asanyarray(img.dataobj)

    if axis is None:
        out_data = np.rollaxis(out_data, 0, out_data.ndim)
    else:
        out_data = np.concatenate(out_data, axis=axis)

    return klass(out_data, affine, header)


def four_to_three(img):
    """Create 3D images from 4D image by slicing over last axis

    Parameters
    ----------
    img :  image
       4D image instance of some class with methods ``get_data``,
       ``header`` and ``affine``, and a class constructor
       allowing klass(data, affine, header)

    Returns
    -------
    imgs : list
       list of 3D images
    """
    arr = np.asanyarray(img.dataobj)
    header = img.header
    affine = img.affine
    image_maker = img.__class__
    if arr.ndim != 4:
        raise ValueError('Expecting four dimensions')
    imgs = []
    for i in range(arr.shape[3]):
        arr3d = arr[..., i]
        img3d = image_maker(arr3d, affine, header)
        imgs.append(img3d)
    return imgs


def as_closest_canonical(img, enforce_diag=False):
    """Return `img` with data reordered to be closest to canonical

    Canonical order is the ordering of the output axes.

    Parameters
    ----------
    img : ``spatialimage``
    enforce_diag : {False, True}, optional
       If True, before transforming image, check if the resulting image
       affine will be close to diagonal, and if not, raise an error

    Returns
    -------
    canonical_img : ``spatialimage``
       Version of `img` where the underlying array may have been
       reordered and / or flipped so that axes 0,1,2 are those axes in
       the input data that are, respectively, closest to the output axis
       orientation.  We modify the affine accordingly.  If `img` is
       already has the correct data ordering, we just return `img`
       unmodified.
    """
    # Get the image class to transform the data for us
    img = img.as_reoriented(io_orientation(img.affine))

    # however, the affine may not be diagonal
    if enforce_diag and not _aff_is_diag(img.affine):
        raise OrientationError('Transformed affine is not diagonal')

    return img


def _aff_is_diag(aff):
    """Utility function returning True if affine is nearly diagonal"""
    rzs_aff = aff[:3, :3]
    return np.allclose(rzs_aff, np.diag(np.diag(rzs_aff)))
