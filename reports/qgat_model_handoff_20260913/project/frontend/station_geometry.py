"""Fixed three-station geometry; visibility is eligibility, not detection recall."""
import numpy as np

def station_positions(width_m, length_m):
    """Training-fit W/L; BS2 is 100m outside the quantile road edge."""
    if not np.isfinite([width_m, length_m]).all() or min(width_m, length_m) <= 0:
        raise ValueError("Road width and length must be positive finite metres")
    return np.array([[-width_m/2-50, -length_m/4],
                     [width_m/2+100, 0],
                     [-width_m/2-50, length_m/4]], dtype=float)

def station_boresights(stations):
    """Left stations face +x; right stations face -x, fixed across scenes."""
    stations = np.asarray(stations, dtype=float)
    if stations.shape != (3, 2) or not np.isfinite(stations).all() or np.any(stations[:, 0] == 0):
        raise ValueError("Expected three finite off-axis station positions")
    return np.where(stations[:, 0] < 0, 0., np.pi)

def visibility(xy, stations, boresights=None, half_angle_deg=70.,
               range_m=(10., 300.), height_difference_m=5.):
    """Return (..., 3) geometric range/FoV mask for planar positions."""
    xy, stations = np.asarray(xy, dtype=float), np.asarray(stations, dtype=float)
    if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
        raise ValueError("Positions must be finite (..., 2) coordinates")
    default_bore = station_boresights(stations)
    bore = default_bore if boresights is None else np.asarray(boresights, dtype=float)
    if bore.shape != (3,) or not np.isfinite(bore).all():
        raise ValueError("Expected three finite boresight angles in radians")
    if not (0 < half_angle_deg <= 90 and 0 <= range_m[0] < range_m[1]):
        raise ValueError("Invalid FoV or range interval")
    delta = xy[..., None, :] - stations
    distance = np.sqrt(np.sum(delta**2, axis=-1) + height_difference_m**2)
    angle = (np.arctan2(delta[..., 1], delta[..., 0]) - bore + np.pi) % (2*np.pi) - np.pi
    return ((distance >= range_m[0]) & (distance <= range_m[1]) &
            (np.abs(angle) <= np.deg2rad(half_angle_deg)))

if __name__ == "__main__":
    stations = station_positions(30, 440)
    assert np.array_equal(stations, [[-65, -110], [115, 0], [-65, 110]])
    assert np.allclose(station_boresights(stations), [0, np.pi, 0])
    mask = visibility([[0, 0], [1000, 0]], stations)
    assert mask[0].all() and not mask[1].any()
    assert not visibility([[125, 0]], stations)[0, 1]
    print("station geometry checks passed")
