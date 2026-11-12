# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Weighted parcellation of CIFTI-2 brain model axes into parcel axes

The :func:`parcellate` function reduces a :class:`~nibabel.cifti2.cifti2_axes.BrainModelAxis`
of a :class:`~nibabel.cifti2.cifti2.Cifti2Image` to a
:class:`~nibabel.cifti2.cifti2_axes.ParcelsAxis` by computing a weighted mean over the
members of each parcel, while leaving all other matrix axes semantically unchanged.

Members are matched to the greyordinates of the source image by their grey matter identity
(surface structure and vertex index, or voxel IJK indices on a compatible volume grid) --
never by their position in the member arrays.
"""

from __future__ import annotations

import numpy as np

from .cifti2 import Cifti2Image
from .cifti2_axes import BrainModelAxis, ParcelsAxis

__all__ = ['parcellate']

# Accumulation is always done in 64-bit float, independently of the source dtype
_ACCUM_DTYPE = np.float64


def parcellate(
    cimg,
    parcels,
    weights,
    axis=-1,
    *,
    nan_policy='propagate',
    fill_value=np.nan,
    chunk_size=1000,
):
    """Weighted parcellation of a CIFTI-2 brain model axis into a parcel axis

    For every position along the other matrix axes each output parcel value is the
    weighted mean of the source greyordinates describing the parcel members:

    ``result[p] = sum(weight[m] * value[m]) / sum(weight[m])``

    Members are matched by grey matter identity: ``(brain structure, vertex)`` pairs for
    surface elements and ``(i, j, k)`` voxel indices on a compatible volume grid for
    volumetric elements. The position of a member in the input :class:`BrainModelAxis`
    is irrelevant. Parcel membership is stored sparsely (one index/weight pair per
    member) and the source data is read in slice-based chunks, so neither a dense
    "number of greyordinates by number of parcels" mapping matrix nor the full source
    matrix is materialized.

    The weights are caller-supplied statistical weights. They are used verbatim; this
    function does not apply, and does not claim to have applied, surface area or voxel
    volume correction.

    Parameters
    ----------
    cimg : :class:`~nibabel.cifti2.cifti2.Cifti2Image`
        Input CIFTI-2 image. The axis selected by ``axis`` must be a
        :class:`~nibabel.cifti2.cifti2_axes.BrainModelAxis`.
    parcels : sequence of (str, :class:`~nibabel.cifti2.cifti2_axes.BrainModelAxis`)
        Ordered ``(parcel name, brain model describing the parcel members)`` pairs.
        Parcels may overlap, i.e. the same grey matter identity may occur in more than
        one parcel, but a grey matter identity may not occur twice within one parcel.
    weights : sequence of 1D array_like
        One weight array per parcel, in the same order as ``parcels``. Each array has
        one finite, non-negative weight per member of the corresponding brain model.
        Every parcel needs at least one strictly positive weight.
    axis : int, optional
        Dimension of ``cimg`` carrying the
        :class:`~nibabel.cifti2.cifti2_axes.BrainModelAxis` to reduce. Defaults to the
        last axis.
    nan_policy : {'propagate', 'omit'}, optional
        How to treat ``NaN`` values in the source data:

        - ``'propagate'`` (default): a ``NaN`` at a member with strictly positive weight
          propagates into the parcel value. Members with zero weight never pollute the
          result, even when their value is ``NaN``.
        - ``'omit'``: ``NaN`` members are skipped and the denominator is recomputed from
          the valid (non-``NaN``) members with positive weight at each position. If no
          valid member remains, ``fill_value`` is returned.
    fill_value : float, optional
        Value returned by the ``'omit'`` policy when all members of a parcel are missing
        at a given position. Defaults to ``numpy.nan``. Ignored for ``'propagate'``.
    chunk_size : int, optional
        Number of rows read at once along the first non-reduced axis. Lower values keep
        the peak memory smaller at the cost of more reads. Defaults to 1000.

    Returns
    -------
    :class:`~nibabel.cifti2.cifti2.Cifti2Image`
        New image with the selected brain model axis replaced by a
        :class:`~nibabel.cifti2.cifti2_axes.ParcelsAxis` recording the exact surface and
        volume members of every parcel in the input parcel order. The other axes and
        their metadata are preserved and the outer NIfTI-2 header is updated to the new
        shape. Data are always returned as 64-bit floating point.

    Raises
    ------
    ValueError
        If the parcel definitions or weights are invalid, if a member cannot be located
        in the source brain model (unknown member, conflicting surface number of
        vertices, or incompatible volume grid), or if an unknown ``nan_policy`` is given.
        All validation errors are raised before any source data is read, so members are
        never silently dropped and weight denominators are never silently changed.
    """
    if nan_policy not in ('propagate', 'omit'):
        raise ValueError(
            f"nan_policy should be 'propagate' or 'omit', not {nan_policy!r}"
        )
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError(f'chunk_size should be a positive integer, not {chunk_size!r}')

    parcels = list(parcels)
    weights = list(weights)
    if len(parcels) != len(weights):
        raise ValueError(
            f'Got {len(parcels)} parcel definitions but {len(weights)} weight arrays'
        )
    if len(parcels) == 0:
        raise ValueError('At least one parcel needs to be defined')

    ndim = cimg.dataobj.ndim
    r = axis % ndim
    src_bm = cimg.header.get_axis(r)
    if not isinstance(src_bm, BrainModelAxis):
        raise ValueError(
            f'Axis {r} is a {type(src_bm).__name__}, not a BrainModelAxis; '
            'parcellation can only reduce a BrainModelAxis'
        )

    # All validation and mapping happens before any source data is read.
    parcel_maps = _build_parcel_maps(src_bm, parcels, weights)
    parcel_axis = ParcelsAxis.from_brain_models(
        [(name, bm) for (name, bm, _, _) in parcel_maps]
    )

    used_ranges, compact_maps = _build_compact_layout(src_bm, parcel_maps)
    n_used = sum(stop - start for start, stop in used_ranges)

    dataobj = cimg.dataobj
    src_shape = tuple(dataobj.shape)
    other_dims = [dim for dim in range(ndim) if dim != r]
    nparcels = len(parcel_maps)
    n_other = int(np.prod([src_shape[dim] for dim in other_dims])) if other_dims else 1

    out_flat = np.empty((n_other, nparcels), dtype=_ACCUM_DTYPE)

    if not other_dims:
        # One-dimensional image: a single chunk with a single "other" position.
        slab = np.empty((1, n_used), dtype=_ACCUM_DTYPE)
        _read_chunk(dataobj, (), r, used_ranges, slab)
        _reduce_chunk(slab, compact_maps, nan_policy, fill_value, out_flat, 0, 1)

    else:
        lead_dim = other_dims[0]
        trail_dims = other_dims[1:]
        lead_len = src_shape[lead_dim]
        n_trail = int(np.prod([src_shape[dim] for dim in trail_dims])) if trail_dims else 1
        # Compact buffer holding only the greyordinates that participate in any parcel.
        max_chunk_rows = min(chunk_size, lead_len) * n_trail
        compact = np.empty((max_chunk_rows, n_used), dtype=_ACCUM_DTYPE)
        flat_start = 0
        for lead_start in range(0, lead_len, chunk_size):
            lead_stop = min(lead_start + chunk_size, lead_len)
            npos = (lead_stop - lead_start) * n_trail
            slab = compact[:npos, :]
            lead_slice = slice(lead_start, lead_stop)
            _read_chunk(
                dataobj, tuple(other_dims), r, used_ranges, slab, lead_dim, lead_slice
            )
            _reduce_chunk(
                slab, compact_maps, nan_policy, fill_value, out_flat, flat_start, npos
            )
            flat_start += npos

    moved_shape = tuple(src_shape[dim] for dim in other_dims) + (nparcels,)
    result = np.ascontiguousarray(
        np.moveaxis(out_flat.reshape(moved_shape), -1, r)
    )

    new_axes = [cimg.header.get_axis(dim) for dim in range(ndim)]
    new_axes[r] = parcel_axis
    new_img = Cifti2Image(result, header=new_axes, nifti_header=cimg.nifti_header)
    # Preserve matrix-level metadata (per-axis metadata travels with the reused axes).
    matrix_meta = cimg.header.matrix.metadata
    if matrix_meta is not None:
        new_img.header.matrix.metadata = matrix_meta
    return new_img


def _build_parcel_maps(src_bm, parcels, weights):
    """Validates parcel definitions and maps every member to a source greyordinate

    Returns a list of ``(name, member_bm, source_indices, member_weights)`` tuples.
    Nothing is read from the source data while building these maps, so all structural
    problems are reported before computation starts.
    """
    # Grey matter identity lookups on the source brain model.
    surface_lookup = {}
    volume_lookup = {}
    for global_idx in range(src_bm.size):
        struct = str(src_bm.name[global_idx])
        if struct in src_bm.nvertices:
            surface_lookup[(struct, int(src_bm.vertex[global_idx]))] = global_idx
        else:
            volume_lookup[tuple(int(v) for v in src_bm.voxel[global_idx])] = global_idx

    src_has_volume = src_bm.affine is not None
    parcel_maps = []
    for (parcel_name, member_bm), member_weights in zip(parcels, weights):
        if not isinstance(member_bm, BrainModelAxis):
            raise TypeError(
                f'Parcel {parcel_name!r} is described by a {type(member_bm).__name__}, '
                'not a BrainModelAxis'
            )
        if member_bm.size == 0:
            raise ValueError(f'Parcel {parcel_name!r} has no members')

        w = np.asarray(member_weights, dtype=_ACCUM_DTYPE)
        if w.shape != (member_bm.size,):
            raise ValueError(
                f'Parcel {parcel_name!r} has {member_bm.size} members but '
                f'{w.size} weights'
            )
        if not np.isfinite(w).all():
            raise ValueError(f'Weights of parcel {parcel_name!r} must be finite')
        if np.any(w < 0):
            raise ValueError(f'Weights of parcel {parcel_name!r} must be non-negative')
        if not np.any(w > 0):
            raise ValueError(
                f'Parcel {parcel_name!r} needs at least one member with positive weight'
            )

        # Reject duplicated grey matter identities within this parcel.
        surface_mask = member_bm.surface_mask
        if np.any(surface_mask):
            surface_ids = list(
                zip(
                    member_bm.name[surface_mask].tolist(),
                    member_bm.vertex[surface_mask].tolist(),
                )
            )
            if len(set(surface_ids)) != len(surface_ids):
                raise ValueError(
                    f'Parcel {parcel_name!r} contains duplicated surface members'
                )
        if np.any(~surface_mask):
            voxel_ids = member_bm.voxel[~surface_mask]
            voxel_tuples = [tuple(int(v) for v in row) for row in voxel_ids]
            if len(set(voxel_tuples)) != len(voxel_tuples):
                raise ValueError(
                    f'Parcel {parcel_name!r} contains duplicated voxel members'
                )

        # Surface number-of-vertices conflicts are reported explicitly.
        for struct, nvertex in member_bm.nvertices.items():
            if struct in src_bm.nvertices:
                if src_bm.nvertices[struct] != nvertex:
                    raise ValueError(
                        f'Surface {struct} has {nvertex} vertices in parcel '
                        f'{parcel_name!r} but {src_bm.nvertices[struct]} in the source '
                        'brain model'
                    )

        # Volume grids have to be the same before voxel IJK identities can be compared.
        if np.any(~surface_mask):
            if not src_has_volume:
                raise ValueError(
                    f'Parcel {parcel_name!r} contains voxels but the source brain model '
                    'does not define a volume'
                )
            if member_bm.volume_shape != src_bm.volume_shape or not np.allclose(
                member_bm.affine, src_bm.affine
            ):
                raise ValueError(
                    f'Parcel {parcel_name!r} is defined on a different volume grid than '
                    'the source brain model'
                )

        source_indices = np.empty(member_bm.size, dtype=np.intp)
        for member_idx in range(member_bm.size):
            if surface_mask[member_idx]:
                identity = (
                    str(member_bm.name[member_idx]),
                    int(member_bm.vertex[member_idx]),
                )
                lookup = surface_lookup
            else:
                identity = tuple(int(v) for v in member_bm.voxel[member_idx])
                lookup = volume_lookup
            try:
                source_indices[member_idx] = lookup[identity]
            except KeyError:
                raise ValueError(
                    f'Member {identity} of parcel {parcel_name!r} is not part of the '
                    'source brain model'
                ) from None

        parcel_maps.append((str(parcel_name), member_bm, source_indices, w))
    return parcel_maps


def _build_compact_layout(src_bm, parcel_maps):
    """Lays out the used source structure ranges into one compact grey matter buffer

    The compact layout only covers source structures that contribute members to at
    least one parcel; unused greyordinates are never read. Returns the source ranges
    to read and, per parcel, the member indices within the compact buffer.
    """
    used = np.zeros(src_bm.size, dtype=bool)
    for _, _, source_indices, _ in parcel_maps:
        used[source_indices] = True

    used_ranges = []
    for _, structure_slice, _ in src_bm.iter_structures():
        start = structure_slice.start
        # iter_structures uses stop=None for the final structure
        stop = src_bm.size if structure_slice.stop is None else structure_slice.stop
        if used[start:stop].any():
            used_ranges.append((start, stop))

    range_starts = np.array([start for start, _ in used_ranges], dtype=np.intp)
    range_stops = np.array([stop for _, stop in used_ranges], dtype=np.intp)
    range_offsets = np.zeros(len(used_ranges), dtype=np.intp)
    if len(used_ranges):
        range_lengths = range_stops - range_starts
        range_offsets[1:] = np.cumsum(range_lengths[:-1])

    compact_maps = []
    for _, _, source_indices, weights in parcel_maps:
        range_idx = np.searchsorted(range_stops, source_indices, side='right')
        compact_indices = (
            source_indices - range_starts[range_idx] + range_offsets[range_idx]
        )
        compact_maps.append((compact_indices, weights))
    return used_ranges, compact_maps


def _read_chunk(
    dataobj, other_dims, reduce_axis, used_ranges, out_buffer, lead_dim=None,
    lead_slice=None,
):
    """Reads the used source ranges for one chunk using only basic (proxy-safe) slices"""
    if lead_dim is None:
        other_slices = {}
    else:
        other_slices = {lead_dim: lead_slice}
    compact_offset = 0
    for start, stop in used_ranges:
        key = [slice(None)] * dataobj.ndim
        for dim in other_dims:
            key[dim] = other_slices.get(dim, slice(None))
        key[reduce_axis] = slice(start, stop)
        slab = np.asarray(dataobj[tuple(key)], dtype=_ACCUM_DTYPE)
        reordered = np.moveaxis(slab, reduce_axis, -1)
        n_columns = stop - start
        out_buffer[:, compact_offset : compact_offset + n_columns] = reordered.reshape(
            -1, n_columns
        )
        compact_offset += n_columns


def _reduce_chunk(
    slab, compact_maps, nan_policy, fill_value, out_flat, row_start, npos
):
    """Computes the weighted parcel means for one chunk of "other axis" positions"""
    rows = slice(row_start, row_start + npos)
    for parcel_idx, (compact_indices, weights) in enumerate(compact_maps):
        values = slab[:, compact_indices]
        weight_sum = weights.sum()  # strictly positive by validation
        nan_mask = np.isnan(values)
        if nan_policy == 'propagate':
            numerator = (np.where(nan_mask, 0.0, values) * weights).sum(axis=1)
            polluted = (nan_mask & (weights > 0)).any(axis=1)
            parcel_values = numerator / weight_sum
            if polluted.any():
                parcel_values = parcel_values.copy()
                parcel_values[polluted] = np.nan
        else:
            valid = ~nan_mask
            denominator = (weights * valid).sum(axis=1)
            numerator = (np.where(valid, values, 0.0) * weights).sum(axis=1)
            safe_denominator = np.where(denominator == 0, 1.0, denominator)
            parcel_values = np.where(
                denominator > 0, numerator / safe_denominator, fill_value
            )
        out_flat[rows, parcel_idx] = parcel_values
