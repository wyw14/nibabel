# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Lazy alignment of CIFTI-2 :class:`BrainModelAxis` axes to a target space

The public entry point is :func:`align_brain_model_axis`.  It reorders one
matrix axis of a :class:`~nibabel.cifti2.cifti2.Cifti2Image` so that its
brainordinates follow a target :class:`BrainModelAxis`, while the data are read
from the source file on demand through :class:`BrainModelAxisProxy`.

Alignment is based on the *identity* of each gray matter element, never on its
current array position:

* a surface element is identified by its brain structure and vertex number;
* a volume element is identified by its ``(i, j, k)`` voxel index on the
  shared volume grid.

No resampling or nearest-point matching is performed.  Axes that do not share a
compatible surface or volume description are rejected up front, before any data
are read, and target elements missing from the source axis are either reported
as an error or filled with a caller-provided value.
"""

from copy import deepcopy
from numbers import Integral

import numpy as np

from ..fileslice import canonical_slicers, is_fancy
from .cifti2 import Cifti2Header, Cifti2Image
from .cifti2_axes import BrainModelAxis

# Index stored in the alignment mapping for target elements absent from the
# source axis.  It must never be used as an array index (where -1 would alias
# the last element); :class:`BrainModelAxisProxy` tests for it explicitly.
MISSING_INDEX = -1


class BrainModelAlignmentError(ValueError):
    """Raised when two :class:`BrainModelAxis` axes cannot be aligned

    Subclasses :class:`ValueError`, so code catching ``ValueError`` keeps
    working.  Typical causes are duplicate gray matter identities, conflicting
    brain-structure assignments for a shared voxel, incompatible volume grids
    or surface sizes, and (with the default ``missing='error'`` policy) target
    elements that are absent from the source axis.
    """


def _element_identities(axis):
    """Return the identity and brain structure of every element of `axis`

    Surface elements use the identity ``(True, structure, vertex)``; volume
    elements use ``(False, i, j, k)``.  The voxel identity deliberately omits
    the brain structure: voxels live on a shared grid, so the same ``(i, j,
    k)`` triple is the same element regardless of the structure it was assigned
    to.

    Parameters
    ----------
    axis : BrainModelAxis

    Returns
    -------
    identities : list of tuples
        One hashable identity per axis element, in axis order.
    structures : list of str
        Brain structure name per axis element, used to detect voxels assigned
        to conflicting brain structures.
    """
    surface_mask = axis.surface_mask
    names = [str(name) for name in axis.name]
    identities = []
    for index, is_surface in enumerate(surface_mask):
        if is_surface:
            identities.append((True, names[index], int(axis.vertex[index])))
        else:
            ijk = axis.voxel[index]
            identities.append((False, int(ijk[0]), int(ijk[1]), int(ijk[2])))
    return identities, names


def _validate_unique_identities(identities, structures, side):
    """Reject duplicate identities and conflicting voxel structure assignments

    Parameters
    ----------
    identities : list of tuples
        Output of :func:`_element_identities`.
    structures : list of str
        Brain structure names parallel to `identities`.
    side : str
        ``'source'`` or ``'target'``, used in error messages.

    Raises
    ------
    BrainModelAlignmentError
        If an identity occurs more than once.  A voxel shared by different
        brain structures is reported as a semantic conflict rather than a plain
        duplicate.
    """
    first_seen = {}
    for index, identity in enumerate(identities):
        previous = first_seen.get(identity)
        if previous is None:
            first_seen[identity] = index
            continue
        if not identity[0] and structures[previous] != structures[index]:
            raise BrainModelAlignmentError(
                f'Voxel {identity[1:]} is assigned to conflicting brain structures '
                f'{structures[previous]!r} and {structures[index]!r} in the {side} axis'
            )
        raise BrainModelAlignmentError(
            f'Duplicate gray matter identity {identity!r} found at positions '
            f'{previous} and {index} in the {side} axis'
        )


def _validate_shared_geometry(source, target, atol, rtol):
    """Check that surfaces and the common volume grid are compatible

    A check is only applied to geometry that both axes define.  Volume shapes
    must match exactly and affines within the given tolerances; surface
    structures present in both axes must declare the same number of vertices.

    Raises
    ------
    BrainModelAlignmentError
        For any incompatibility.  No implicit resampling is attempted.
    """
    source_has_volume = bool((~source.surface_mask).any())
    target_has_volume = bool((~target.surface_mask).any())
    if source_has_volume and target_has_volume:
        if source.volume_shape != target.volume_shape:
            raise BrainModelAlignmentError(
                f'Volume shapes differ: {source.volume_shape} versus {target.volume_shape}'
            )
        if not np.allclose(source.affine, target.affine, atol=atol, rtol=rtol):
            raise BrainModelAlignmentError(
                'Volume affines differ by more than the tolerated amount '
                f'(atol={atol}, rtol={rtol})'
            )
    for structure, nvertices in target.nvertices.items():
        source_nvertices = source.nvertices.get(structure)
        if source_nvertices is not None and source_nvertices != nvertices:
            raise BrainModelAlignmentError(
                f'Surface {structure!r} has {source_nvertices} vertices in the source axis '
                f'but {nvertices} in the target axis; resampling is not supported'
            )


def compute_brain_model_mapping(source, target, *, atol=1e-5, rtol=1e-5):
    """Build an index mapping aligning `target` brainordinates onto `source`

    The returned array has one entry per target element giving the position of
    the matching element in the source axis, or :data:`MISSING_INDEX` (``-1``)
    when the target element is absent from the source.  All validation
    (duplicate identities, voxel structure conflicts, shared geometry) happens
    here, before any matrix data are read.

    Parameters
    ----------
    source : BrainModelAxis
        Axis describing the brainordinates of the source matrix.
    target : BrainModelAxis
        Axis describing the desired brainordinate order.
    atol : float, optional
        Absolute tolerance for comparing shared volume affines.
    rtol : float, optional
        Relative tolerance for comparing shared volume affines.

    Returns
    -------
    mapping : numpy.ndarray of intp
        Array of shape ``(len(target),)`` with source indices or -1.
    missing : list of tuple
        ``(target_index, identity)`` pairs for every target element without a
        match in the source axis.  Empty when every element was found.

    Raises
    ------
    TypeError
        If either argument is not a :class:`BrainModelAxis`.
    BrainModelAlignmentError
        If the axes cannot be aligned for any of the reasons documented on
        :class:`BrainModelAlignmentError`.
    """
    if not isinstance(source, BrainModelAxis) or not isinstance(target, BrainModelAxis):
        raise TypeError(
            'source and target must both be BrainModelAxis instances, got '
            f'{type(source).__name__} and {type(target).__name__}'
        )
    _validate_shared_geometry(source, target, atol=atol, rtol=rtol)
    source_ids, source_structures = _element_identities(source)
    target_ids, target_structures = _element_identities(target)
    _validate_unique_identities(source_ids, source_structures, 'source')
    _validate_unique_identities(target_ids, target_structures, 'target')

    source_lookup = {identity: index for index, identity in enumerate(source_ids)}
    mapping = np.full(len(target), MISSING_INDEX, dtype=np.intp)
    missing = []
    for index, identity in enumerate(target_ids):
        source_index = source_lookup.get(identity)
        if source_index is None:
            missing.append((index, identity))
        else:
            mapping[index] = source_index
    return mapping, missing


def _contiguous_ranges(indices):
    """Group a sorted integer array into ``(start, stop)`` contiguous ranges"""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [indices.size - 1]))
    return [(int(indices[start]), int(indices[end]) + 1) for start, end in zip(starts, ends)]


class BrainModelAxisProxy:
    """Lazy, reordered view of a CIFTI matrix along one :class:`BrainModelAxis`

    The proxy wraps the ``dataobj`` of a source CIFTI-2 image together with a
    precomputed index mapping.  Slicing the proxy only reads the source rows or
    columns required for the requested output: needed source indices are first
    grouped into contiguous ranges, each range is read with a single basic
    slice, and the values are then scattered back into target order.  The full
    source matrix is never materialized, and empty or fully missing outputs do
    not touch the source data at all.

    Parameters
    ----------
    dataobj : array-like
        Source matrix data object, typically the ``dataobj`` of a loaded
        :class:`~nibabel.cifti2.cifti2.Cifti2Image`.  Only basic slicing is
        required of it.
    mapping : numpy.ndarray of int
        1D array mapping each target position to a source position, with -1
        marking a missing element.  A defensive copy is taken.
    axis : int
        Matrix dimension described by the mapping.  Negative values count from
        the last dimension.
    fill_value : scalar, optional
        Value used for missing elements.  It participates in output dtype
        promotion whenever the mapping contains missing positions.

    Attributes
    ----------
    shape : tuple of int
        Matrix shape with the aligned dimension replaced by the target size.
    dtype : numpy.dtype
        Output dtype; promoted with `fill_value` when missing positions exist.
    is_proxy : bool
        Always ``True``.
    """

    is_proxy = True

    def __init__(self, dataobj, mapping, axis, fill_value=0):
        mapping = np.asarray(mapping, dtype=np.intp)
        if mapping.ndim != 1:
            raise ValueError('The alignment mapping must be one-dimensional')
        source_shape = tuple(dataobj.shape)
        ndim = len(source_shape)
        if ndim == 0:
            raise BrainModelAlignmentError('Cannot align a scalar matrix')
        axis = int(axis)
        if not -ndim <= axis < ndim:
            raise np.exceptions.AxisError(axis, ndim)
        axis = axis % ndim
        source_axis_size = source_shape[axis]
        if np.any(mapping < MISSING_INDEX) or np.any(mapping >= source_axis_size):
            raise ValueError(
                'The alignment mapping contains source indices outside the source matrix '
                f'(axis size {source_axis_size})'
            )

        self._dataobj = dataobj
        # Defensive copy: later mutation of the caller's array must not affect
        # reads through this proxy.
        self._mapping = mapping.copy()
        self._axis = axis
        self._shape = source_shape[:axis] + (mapping.size,) + source_shape[axis + 1 :]
        self._has_missing = bool(np.any(self._mapping == MISSING_INDEX))
        self._fill_value = np.asarray(fill_value)
        if self._has_missing:
            self._dtype = np.dtype(np.result_type(np.dtype(dataobj.dtype), self._fill_value.dtype))
        else:
            self._dtype = np.dtype(dataobj.dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def mapping(self):
        """Read-only view of the target-to-source index mapping"""
        view = self._mapping.view()
        view.setflags(write=False)
        return view

    def __len__(self):
        return self._shape[0]

    def __array__(self, dtype=None):
        array = np.asanyarray(self[()])
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    @staticmethod
    def _insert_new_axes(array, new_axis_positions):
        """Re-insert ``np.newaxis`` dimensions stripped before reading"""
        for position in sorted(new_axis_positions, reverse=True):
            array = np.expand_dims(array, position)
        return array

    def __getitem__(self, slicer):
        if not isinstance(slicer, tuple):
            slicer = (slicer,)
        if is_fancy(slicer):
            raise ValueError(
                'BrainModelAxisProxy only supports basic slicing '
                '(integers, slices, ellipsis and np.newaxis)'
            )
        canonical = canonical_slicers(slicer, self._shape, check_inds=True)
        # np.newaxis entries do not address a real dimension; remember where to
        # re-insert them in the output and process real dimensions only.
        new_axis_positions = tuple(i for i, entry in enumerate(canonical) if entry is None)
        real_slicers = tuple(entry for entry in canonical if entry is not None)
        axis = self._axis
        aligned_slicer = real_slicers[axis]

        out_shape = tuple(
            len(range(*entry.indices(self._shape[dim])))
            for dim, entry in enumerate(real_slicers)
            if not isinstance(entry, Integral)
        )

        # Source indices needed for this request, in requested target order.
        if isinstance(aligned_slicer, Integral):
            aligned_scalar = True
            needed = self._mapping[aligned_slicer : aligned_slicer + 1]
        else:
            aligned_scalar = False
            needed = self._mapping[aligned_slicer]

        # Output dimension carrying the aligned axis, after integer axes collapse.
        out_aligned_axis = sum(not isinstance(entry, Integral) for entry in real_slicers[:axis])

        # Empty selection: nothing to read.
        if needed.size == 0:
            return self._insert_new_axes(
                np.empty(out_shape, dtype=self._dtype), new_axis_positions
            )

        valid = needed > MISSING_INDEX
        # Every requested brainordinate is missing: fill without any source read.
        if not valid.any():
            result = np.full(out_shape, self._fill_value, dtype=self._dtype)
            return self._insert_new_axes(result, new_axis_positions)

        result = (
            np.empty(out_shape, dtype=self._dtype)
            if valid.all()
            else np.full(out_shape, self._fill_value, dtype=self._dtype)
        )

        if aligned_scalar:
            # The aligned dimension collapses; the single requested brainordinate
            # is valid (missing case was handled above).
            source_slicer = tuple(
                int(needed[0]) if dim == axis else entry for dim, entry in enumerate(real_slicers)
            )
            result[...] = self._dataobj[source_slicer]
            return self._insert_new_axes(result, new_axis_positions)

        # Read each contiguous run of required source indices once, then scatter
        # the values into the requested target positions (which may be reversed).
        for start, stop in _contiguous_ranges(np.unique(needed[valid])):
            source_slicer = tuple(
                slice(start, stop) if dim == axis else entry
                for dim, entry in enumerate(real_slicers)
            )
            chunk = self._dataobj[source_slicer]
            in_range = valid & (needed >= start) & (needed < stop)
            output_positions = np.flatnonzero(in_range)
            local_positions = needed[output_positions] - start
            output_index = tuple(
                output_positions if dim == out_aligned_axis else slice(None)
                for dim in range(result.ndim)
            )
            chunk_index = tuple(
                local_positions if dim == out_aligned_axis else slice(None)
                for dim in range(result.ndim)
            )
            result[output_index] = chunk[chunk_index]
        return self._insert_new_axes(result, new_axis_positions)


def _format_missing(missing_elements, limit=5):
    examples = ', '.join(repr(identity) for _, identity in missing_elements[:limit])
    if len(missing_elements) > limit:
        examples += ', ...'
    return examples


def align_brain_model_axis(
    image, target, axis=-1, *, missing='error', fill_value=0, atol=1e-5, rtol=1e-5
):
    """Align one :class:`BrainModelAxis` of a CIFTI-2 image to `target`

    Returns a new :class:`~nibabel.cifti2.cifti2.Cifti2Image` whose selected
    matrix dimension follows the element order of the `target` brain model
    axis.  Its ``dataobj`` is a :class:`BrainModelAxisProxy`, so source data
    are only read when the result is sliced or converted to an array; the
    source image itself is never cached or fully materialized.

    Alignment uses gray matter identities (structure + vertex for surfaces,
    shared-grid ``(i, j, k)`` for voxels), not array positions.  The other
    matrix dimensions keep their original axes (e.g. series or scalar
    metadata), and the returned :class:`~nibabel.cifti2.cifti2.Cifti2Header` is
    built from scratch through :meth:`~nibabel.cifti2.cifti2.Cifti2Header.from_axes`
    rather than by editing the source XML.  The NIfTI-2 intent is preserved and
    the outer dimensions are updated to match the new matrix shape.

    Parameters
    ----------
    image : Cifti2Image
        Source image.  Only its ``dataobj``, header axes and NIfTI-2 header are
        used; its data caches are not populated.
    target : BrainModelAxis
        Desired order of brainordinates along `axis`.
    axis : int, optional
        Matrix dimension to align, defaulting to the last dimension.  Negative
        values count from the last dimension.  The current axis at this
        position must itself be a :class:`BrainModelAxis`.
    missing : {'error', 'fill'}, optional
        Policy for target elements absent from the source axis.  ``'error'``
        (the default) raises :class:`BrainModelAlignmentError`; ``'fill'``
        inserts `fill_value` instead.
    fill_value : scalar, optional
        Fill value used when ``missing='fill'``.  When at least one element is
        missing, the output (and NIfTI-2 storage) dtype is promoted to hold it.
    atol : float, optional
        Absolute tolerance for comparing shared volume affines.
    rtol : float, optional
        Relative tolerance for comparing shared volume affines.

    Returns
    -------
    aligned : Cifti2Image
        Image with a :class:`BrainModelAxisProxy` as ``dataobj`` and a freshly
        constructed header based on `target`.  All axis metadata are snapshots:
        later modification of the input axes does not affect the result.

    Raises
    ------
    BrainModelAlignmentError
        For duplicate identities, conflicting voxel structure assignments,
        incompatible volume grids or surface sizes, a non-brain-model `axis`,
        or missing target elements when ``missing='error'``.
    TypeError
        If `image` is not a :class:`~nibabel.cifti2.cifti2.Cifti2Image` or
        `target` is not a :class:`BrainModelAxis`.
    ValueError
        If `missing` is not one of ``'error'`` or ``'fill'``.
    """
    if not isinstance(image, Cifti2Image):
        raise TypeError(f'image must be a Cifti2Image, got {type(image).__name__}')
    if not isinstance(target, BrainModelAxis):
        raise TypeError(f'target must be a BrainModelAxis, got {type(target).__name__}')
    if missing not in ('error', 'fill'):
        raise ValueError("missing must be 'error' or 'fill'")
    ndim = len(image.shape)
    if ndim == 0:
        raise BrainModelAlignmentError('Cannot align a scalar CIFTI-2 matrix')
    aligned_axis = int(axis)
    if not -ndim <= aligned_axis < ndim:
        raise np.exceptions.AxisError(aligned_axis, ndim)
    aligned_axis %= ndim

    source_axis = image.header.get_axis(aligned_axis)
    if not isinstance(source_axis, BrainModelAxis):
        raise BrainModelAlignmentError(
            f'Axis {aligned_axis} is a {type(source_axis).__name__}, not a BrainModelAxis'
        )

    mapping, missing_elements = compute_brain_model_mapping(
        source_axis, target, atol=atol, rtol=rtol
    )
    if missing == 'error' and missing_elements:
        raise BrainModelAlignmentError(
            f'{len(missing_elements)} target element(s) are not present in the source '
            f'axis; use missing="fill" to replace them. First missing identities: '
            f'{_format_missing(missing_elements)}'
        )

    # Snapshot every axis feeding the new header so later caller-side mutation
    # cannot change the constructed image.
    new_axes = tuple(
        deepcopy(target) if dim == aligned_axis else deepcopy(image.header.get_axis(dim))
        for dim in range(ndim)
    )
    header = Cifti2Header.from_axes(new_axes)
    proxy = BrainModelAxisProxy(image.dataobj, mapping, aligned_axis, fill_value=fill_value)
    aligned = Cifti2Image(proxy, header=header, nifti_header=image.nifti_header.copy())
    # The fill value may promote the output dtype; keep the NIfTI-2 header honest.
    aligned.set_data_dtype(proxy.dtype)
    return aligned
