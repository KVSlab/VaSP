# Chroma Filterbanks - Code adapted from librosa
#
# ISC License
# Copyright (c) 2013--2023, librosa development team.
#
# Permission to use, copy, modify, and/or distribute this software for any purpose
# with or without fee is hereby granted, provided that the above copyright notice
# and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
# TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
# THIS SOFTWARE.

import numpy as np


class ParameterError(Exception):
    """Exception class for mal-formed inputs"""

    pass


def octs_to_hz(octs, tuning=0.0, bins_per_octave=12):
    """Convert octaves numbers to frequencies.

    Octaves are counted relative to A.

    Examples
    --------
    >>> librosa.octs_to_hz(1)
    55.
    >>> librosa.octs_to_hz([-2, -1, 0, 1, 2])
    array([   6.875,   13.75 ,   27.5  ,   55.   ,  110.   ])

    Parameters
    ----------
    octaves       : np.ndarray [shape=(n,)] or float
        octave number for each frequency

    tuning : float
        Tuning deviation from A440 in (fractional) bins per octave.

    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    frequencies   : number or np.ndarray [shape=(n,)]
        scalar or vector of frequencies

    See Also
    --------
    hz_to_octs
    """
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)

    return (float(A440) / 16) * (2.0 ** np.asanyarray(octs))


def hz_to_octs(frequencies, tuning=0.0, bins_per_octave=12):
    """Convert frequencies (Hz) to (fractional) octave numbers.

    Examples
    --------
    >>> librosa.hz_to_octs(440.0)
    4.
    >>> librosa.hz_to_octs([32, 64, 128, 256])
    array([ 0.219,  1.219,  2.219,  3.219])

    Parameters
    ----------
    frequencies   : number >0 or np.ndarray [shape=(n,)] or float
        scalar or vector of frequencies

    tuning        : float
        Tuning deviation from A440 in (fractional) bins per octave.

    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    octaves       : number or np.ndarray [shape=(n,)]
        octave number for each frequency

    See Also
    --------
    octs_to_hz
    """

    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)

    return np.log2(np.asanyarray(frequencies) / (float(A440) / 16))


def tiny(x):
    """Compute the tiny-value corresponding to an input's data type.

    This is the smallest "usable" number representable in ``x.dtype``
    (e.g., float32).

    This is primarily useful for determining a threshold for
    numerical underflow in division or multiplication operations.

    Parameters
    ----------
    x : number or np.ndarray
        The array to compute the tiny-value for.
        All that matters here is ``x.dtype``

    Returns
    -------
    tiny_value : float
        The smallest positive usable number for the type of ``x``.
        If ``x`` is integer-typed, then the tiny value for ``np.float32``
        is returned instead.

    See Also
    --------
    numpy.finfo

    Examples
    --------

    For a standard double-precision floating point number:

    >>> librosa.util.tiny(1.0)
    2.2250738585072014e-308

    Or explicitly as double-precision

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float64))
    2.2250738585072014e-308

    Or complex numbers

    >>> librosa.util.tiny(1j)
    2.2250738585072014e-308

    Single-precision floating point:

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float32))
    1.1754944e-38

    Integer

    >>> librosa.util.tiny(5)
    1.1754944e-38
    """

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    """Normalize an array along a chosen axis.

    Given a norm (described below) and a target axis, the input
    array is scaled so that::

        norm(S, axis=axis) == 1

    For example, ``axis=0`` normalizes each column of a 2-d array
    by aggregating over the rows (0-axis).
    Similarly, ``axis=1`` normalizes each row of a 2-d array.

    This function also supports thresholding small-norm slices:
    any slice (i.e., row or column) with norm below a specified
    ``threshold`` can be left un-normalized, set to all-zeros, or
    filled with uniform non-zero values that normalize to 1.

    Note: the semantics of this function differ from
    `scipy.linalg.norm` in two ways: multi-dimensional arrays
    are supported, but matrix-norms are not.


    Parameters
    ----------
    S : np.ndarray
        The matrix to normalize

    norm : {np.inf, -np.inf, 0, float > 0, None}
        - `np.inf`  : maximum absolute value
        - `-np.inf` : mininum absolute value
        - `0`    : number of non-zeros (the support)
        - float  : corresponding l_p norm
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed

    axis : int [scalar]
        Axis along which to compute the norm.

    threshold : number > 0 [optional]
        Only the columns (or rows) with norm at least ``threshold`` are
        normalized.

        By default, the threshold is determined from
        the numerical precision of ``S.dtype``.

    fill : None or bool
        If None, then columns (or rows) with norm below ``threshold``
        are left as is.

        If False, then columns (rows) with norm below ``threshold``
        are set to 0.

        If True, then columns (rows) with norm below ``threshold``
        are filled uniformly such that the corresponding norm is 1.

        .. note:: ``fill=True`` is incompatible with ``norm=0`` because
            no uniform vector exists with l0 "norm" equal to 1.

    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized array

    Raises
    ------
    ParameterError
        If ``norm`` is not among the valid types defined above

        If ``S`` is not finite

        If ``fill=True`` and ``norm=0``

    See Also
    --------
    scipy.linalg.norm

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct an example matrix
    >>> S = np.vander(np.arange(-2.0, 2.0))
    >>> S
    array([[-8.,  4., -2.,  1.],
           [-1.,  1., -1.,  1.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.]])
    >>> # Max (l-infinity)-normalize the columns
    >>> librosa.util.normalize(S)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # Max (l-infinity)-normalize the rows
    >>> librosa.util.normalize(S, axis=1)
    array([[-1.   ,  0.5  , -0.25 ,  0.125],
           [-1.   ,  1.   , -1.   ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 1.   ,  1.   ,  1.   ,  1.   ]])
    >>> # l1-normalize the columns
    >>> librosa.util.normalize(S, norm=1)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    >>> # l2-normalize the columns
    >>> librosa.util.normalize(S, norm=2)
    array([[-0.985,  0.943, -0.816,  0.5  ],
           [-0.123,  0.236, -0.408,  0.5  ],
           [ 0.   ,  0.   ,  0.   ,  0.5  ],
           [ 0.123,  0.236,  0.408,  0.5  ]])

    >>> # Thresholding and filling
    >>> S[:, -1] = 1e-308
    >>> S
    array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
              1.000e-308],
           [ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.000e+000,   1.000e+000,   1.000e+000,
              1.000e-308]])

    >>> # By default, small-norm columns are left untouched
    >>> librosa.util.normalize(S)
    array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [ -1.250e-001,   2.500e-001,  -5.000e-001,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.250e-001,   2.500e-001,   5.000e-001,
              1.000e-308]])
    >>> # Small-norm columns can be zeroed out
    >>> librosa.util.normalize(S, fill=False)
    array([[-1.   ,  1.   , -1.   ,  0.   ],
           [-0.125,  0.25 , -0.5  ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.125,  0.25 ,  0.5  ,  0.   ]])
    >>> # Or set to constant with unit-norm
    >>> librosa.util.normalize(S, fill=True)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # With an l1 norm instead of max-norm
    >>> librosa.util.normalize(S, norm=1, fill=True)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    """

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ParameterError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise ParameterError("Unsupported norm: {}".format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def chroma_filterbank(sr, n_fft, n_chroma=12, tuning=0.0, ctroct=5.0,
                      octwidth=2, norm=2, base_c=True, dtype=np.float32):
    """Create a chroma filter bank.

    This creates a linear transformation matrix to project
    FFT bins onto chroma bins (i.e. pitch classes).


    Parameters
    ----------
    sr        : number > 0 [scalar]
        audio sampling rate

    n_fft     : int > 0 [scalar]
        number of FFT bins

    n_chroma  : int > 0 [scalar]
        number of chroma bins

    tuning : float
        Tuning deviation from A440 in fractions of a chroma bin.

    ctroct    : float > 0 [scalar]

    octwidth  : float > 0 or None [scalar]
        ``ctroct`` and ``octwidth`` specify a dominance window:
        a Gaussian weighting centered on ``ctroct`` (in octs, A0 = 27.5Hz)
        and with a gaussian half-width of ``octwidth``.

        Set ``octwidth`` to `None` to use a flat weighting.

    norm : float > 0 or np.inf
        Normalization factor for each filter

    base_c : bool
        If True, the filter bank will start at 'C'.
        If False, the filter bank will start at 'A'.

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    wts : ndarray [shape=(n_chroma, 1 + n_fft / 2)]
        Chroma filter matrix

    See Also
    --------
    librosa.util.normalize
    librosa.feature.chroma_stft

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Build a simple chroma filter bank

    >>> chromafb = librosa.filters.chroma(22050, 4096)
    array([[  1.689e-05,   3.024e-04, ...,   4.639e-17,   5.327e-17],
           [  1.716e-05,   2.652e-04, ...,   2.674e-25,   3.176e-25],
    ...,
           [  1.578e-05,   3.619e-04, ...,   8.577e-06,   9.205e-06],
           [  1.643e-05,   3.355e-04, ...,   1.474e-10,   1.636e-10]])

    Use quarter-tones instead of semitones

    >>> librosa.filters.chroma(22050, 4096, n_chroma=24)
    array([[  1.194e-05,   2.138e-04, ...,   6.297e-64,   1.115e-63],
           [  1.206e-05,   2.009e-04, ...,   1.546e-79,   2.929e-79],
    ...,
           [  1.162e-05,   2.372e-04, ...,   6.417e-38,   9.923e-38],
           [  1.180e-05,   2.260e-04, ...,   4.697e-50,   7.772e-50]])


    Equally weight all octaves

    >>> librosa.filters.chroma(22050, 4096, octwidth=None)
    array([[  3.036e-01,   2.604e-01, ...,   2.445e-16,   2.809e-16],
           [  3.084e-01,   2.283e-01, ...,   1.409e-24,   1.675e-24],
    ...,
           [  2.836e-01,   3.116e-01, ...,   4.520e-05,   4.854e-05],
           [  2.953e-01,   2.888e-01, ...,   7.768e-10,   8.629e-10]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(chromafb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Chroma filter', title='Chroma filter bank')
    >>> fig.colorbar(img, ax=ax)
    """

    wts = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    frqbins = n_chroma * hz_to_octs(
        frequencies, tuning=tuning, bins_per_octave=n_chroma
    )

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))

    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))

    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype="d")).T

    n_chroma2 = np.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are positive
    D = np.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2 * D / np.tile(binwidthbins, (n_chroma, 1))) ** 2)

    # normalize each column
    wts = normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2)),
            (n_chroma, 1),
        )

    if base_c:
        wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(wts[:, : int(1 + n_fft / 2)], dtype=dtype)
