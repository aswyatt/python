from typing import Final, Callable
from numpy.typing import ArrayLike, NDArray
import numpy as np
from scipy import special, constants, fft

eps0: Final[float] = constants.epsilon_0
me: Final[float] = constants.m_e
e: Final[float] = constants.e
c: Final[float] = constants.c
hbar: Final[float] = constants.hbar

sqrt: Final[Callable] = np.sqrt
exp: Final[Callable] = np.exp
ln: Final[Callable] = np.log

PI: Final[float] = np.pi
TAU: Final[float] = 2 * PI
ROOT2: Final[float] = sqrt(2).item()
INF: Final[float] = np.inf

LIM = np.array([-1, 1])


# Allow one to stop execution at a particular point
def STOP():
    raise SystemExit


# Convert data to a row vector: size=(1,N)
def row(x: NDArray) -> NDArray:
    return x.reshape((1, -1))


# Convert data to a column vector: size=(N, 1)
def col(x: NDArray) -> NDArray:
    return x.reshape((-1, 1))


# Convert data to a "page": size=(N,1,1)
def page(x: NDArray) -> NDArray:
    return x.reshape((-1, 1, 1))


# Interleaves a set of lists
def interleave(lists: list | tuple) -> list:
    return [item for pair in zip(*lists) for item in pair]


def deg2rad(d: ArrayLike) -> ArrayLike:
    ret = np.deg2rad(d % 360)
    return ret.item() if np.isscalar(d) else ret


def rad2deg(r: ArrayLike) -> ArrayLike:
    ret = np.rad2deg(r % TAU)
    return ret.item() if np.isscalar(r) else ret


# Calculate absolute square
def abs2(x: ArrayLike | ArrayLike | ArrayLike) -> ArrayLike:
    return x.real**2 + x.imag**2  # type: ignore


# Normalise by peak value along given axis
def NORM(
    x: ArrayLike, axis: ArrayLike | None = None, fun: Callable = np.max
) -> ArrayLike | ArrayLike:
    return x / fun(x, axis=axis, keepdims=True)


# Normalise by ArrayLikeensity ArrayLikeegral
def NORM2(x: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    return x / np.linalg.norm(x, axis)  # type: ignore


def gauss1d(
    x: ArrayLike,
    x0: ArrayLike = 0.0,
    FWHM: ArrayLike = 1.0,
    order: ArrayLike | ArrayLike = 1,
) -> ArrayLike:
    order = abs(order) * 2  # type: ignore
    X = -ln(2) * np.power(2 * (x - x0) / FWHM, order)
    return exp(X)


# Convert photon energy [eV] <--> wavelength [nm]
def convert_el(x: ArrayLike) -> ArrayLike:
    return (1e9 * constants.h * constants.c / constants.e) / x


# Convert ang. freq [rad/fs] <--> wavelength [nm]
def convert_wl(x: ArrayLike) -> ArrayLike:
    return (TAU * c * 1e-6) / x


# Convert Intensity [TW/cm^2] --> Real Field [V/nm]
def intensity_to_field(I: ArrayLike, ref_index: ArrayLike = 1.0) -> ArrayLike:
    return 0.1 * sqrt(2 * I / (ref_index * eps0 * c))


# Convert Real Field [V/nm] --> Intensity [TW/cm^2]
def field_to_intensity(E: ArrayLike, ref_index: ArrayLike = 1.0) -> ArrayLike:
    return 50 * ref_index * eps0 * c * abs2(E)


# Convert E(t) <--> E(w)
def freq2time(E: NDArray, N: int | None = None, axis: int | None = None) -> NDArray:
    if axis is None:
        axis = E.ndim - 1
    return fft.fftshift(fft.fft(E, N, axis=axis), axes=axis)  # type: ignore


def time2freq(E: NDArray, axis: int | None = None) -> NDArray:
    if axis is None:
        axis = E.ndim - 1
    return fft.ifft(fft.ifftshift(E, axes=axis), axis=axis)  # type: ignore


# Convert E(x) <--> E(kx)
def freq2space(E: NDArray, axis: int | None = None) -> NDArray:
    if axis is None:
        axis = E.ndim - 1
    return fft.fftshift(fft.ifft(fft.ifftshift(E, axes=axis), axis=axis), axes=axis)  # type: ignore


def space2freq(E: NDArray, axis: int | None = None) -> NDArray:
    if axis is None:
        axis = E.ndim - 1
    return fft.fftshift(fft.fft(fft.ifftshift(E, axes=axis), axis=axis), axes=axis)  # type: ignore


# Convert E(x, y) <--> E(kx, ky)
def freq2space2(E: NDArray, axes: int | None = None) -> NDArray:
    if axes is None:
        axes = (E.ndim - 2, E.ndim - 1)
    return fft.fftshift(fft.ifft2(fft.ifftshift(E, axes=axes), axes=axes), axes=axes)  # type: ignore


def space2freq2(E: NDArray, axes: int | None = None) -> NDArray:
    if axes is None:
        axes = (E.ndim - 2, E.ndim - 1)
    return fft.fftshift(fft.fft2(fft.ifftshift(E, axes=axes), axes=axes), axes=axes)  # type: ignore


def weighted_linear(
    x: ArrayLike, y: ArrayLike, w: ArrayLike = 1.0, axis: int = 0, *args, **kwargs
) -> tuple:
    KeepDims = kwargs.pop("keepdims", True)
    SUM = lambda x: np.sum(x, axis=axis, keepdims=KeepDims, *args, **kwargs)
    S = SUM(w)
    Sx = SUM(w * x)
    Sxx = SUM(w * np.power(x, 2))
    Sy = SUM(w * y)
    Sxy = SUM(w * x * y)
    D = S * Sxx - Sx**2
    m = (S * Sxy - Sx * Sy) / D
    c = (Sxx * Sy - Sx * Sxy) / D

    ret = (m, c)

    return ret
