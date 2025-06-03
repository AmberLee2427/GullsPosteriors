import numpy as np
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy import units as u
from astroquery.jplhorizons import Horizons
from scipy.interpolate import interp1d


class Orbit:

    def __init__(self, 
                 obs_location='SEMB-L2', 
                 start_date='2018-04-25',
                 end_date='2023-05-30', 
                 refplane='ecliptic', 
                 n_epochs=None,
                 origin='500@10',
                 date_format='iso'):
        '''Position file from JPL Horizons'''

        self.start_time = Time(start_date, format=date_format)
        self.end_time = Time(end_date, format=date_format)

        if n_epochs is None:
            self.n_epochs = int(self.end_time.jd - self.start_time.jd + 1)  # 1 epoch per day
        else:
            self.n_epochs = n_epochs

        self.obs_location = obs_location
        self.origin = origin
        self.refplane = refplane

        self.epochs, self.positions, self.velocities = self.fetch_horizons_data()

    def fetch_horizons_data(self):
        """Query JPL Horizons for observatory ephemerides.

        This method does not take any arguments and uses the instance
        attributes configured at initialisation.

        Returns
        -------
        tuple(Time, CartesianRepresentation, CartesianDifferential)
            Times, positions and velocities describing the observatory
            trajectory.
        """
        times = np.linspace(self.start_time.jd, self.end_time.jd, self.n_epochs)
        times = Time(times, format='jd')

        positions_list = []
        velocities_list = []

        # Split the times into chunks to avoid hitting the API limits
        chunk_size = 75  # Adjust this based on the API limits
        for i in range(0, len(times), chunk_size):
            chunk_times = times[i:i + chunk_size]
            q = Horizons(id=self.obs_location, location=self.origin, epochs=chunk_times.jd)
            data = q.vectors(refplane=self.refplane)

            positions_list.append(CartesianRepresentation(data['x'], data['y'], data['z']))
            velocities_list.append(CartesianDifferential(data['vx'], data['vy'], data['vz']))

        # Combine the chunks into single arrays
        positions = CartesianRepresentation(
            np.concatenate([pos.x for pos in positions_list]),
            np.concatenate([pos.y for pos in positions_list]),
            np.concatenate([pos.z for pos in positions_list])
        )
        velocities = CartesianDifferential(
            np.concatenate([vel.d_x for vel in velocities_list]),
            np.concatenate([vel.d_y for vel in velocities_list]),
            np.concatenate([vel.d_z for vel in velocities_list])
        )

        return times, positions, velocities

    def get_pos(self, t):
        '''get the position of the observatory at time t by interpolating the position file'''
        t = Time(t, format='jd')
        x_interp = interp1d(self.epochs.jd, self.positions.x.to(u.au).value)
        y_interp = interp1d(self.epochs.jd, self.positions.y.to(u.au).value)
        z_interp = interp1d(self.epochs.jd, self.positions.z.to(u.au).value)
        x = x_interp(t.jd)
        y = y_interp(t.jd)
        z = z_interp(t.jd)
        xyz = np.vstack((x, y, z))
        return xyz
    
    def get_vel(self, t):
        '''get the velocity of the observatory at time t by interpolating the position file'''
        t = Time(t, format='jd')
        vx_interp = interp1d(self.epochs.jd, self.velocities.d_x.to(u.au / u.day).value)
        vy_interp = interp1d(self.epochs.jd, self.velocities.d_y.to(u.au / u.day).value)
        vz_interp = interp1d(self.epochs.jd, self.velocities.d_z.to(u.au / u.day).value)
        vx = vx_interp(t.jd)
        vy = vy_interp(t.jd)
        vz = vz_interp(t.jd)
        vxyz = np.vstack((vx, vy, vz)) * (u.au / u.day)
        return vxyz
