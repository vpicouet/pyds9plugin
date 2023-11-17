

import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_moon, SkyCoord
from astropy.time import Time
from astroplan import Observer, FixedTarget, time_grid_from_range, moon_illumination
import pandas as pd

ra=[22.42638]
dec = [0.622]
hours=[3]
utcoffset = 6.0*u.hour

F1={"RA":32.19,"DEC":-5.688,"hour":4+6,"name":"F1"}
TBS={"RA":274.3341,"DEC":68.6707,"hour":19+6-24,"name":"TBS"}
BQSO2={"RA":235.22,"DEC":47.75,"hour":21+6-24,"name":"BQSO2"}
BQSO1={"RA":275.48,"DEC":64.36,"hour":22+6-24,"name":"BQSO1"}
F2={"RA":253.0624	,"DEC":34.9699,"hour":21+6-24,"name":"F2"}
BQSO3={"RA":11.87,"DEC":3.24,"hour":1+6,"name":"BQSO3"}
QSO2={"RA":22.42638,"DEC":0.622,"hour":3+6,"name":"QSO2"}
QSO3={"RA":14.62883	,"DEC":0.00512,"hour":4+6,"name":"QSO3"}
QSO1={"RA":0.03894,"DEC":1.39459,"hour":2+6,"name":"QSO1"}
F4={"RA":36.9049,"DEC":0.65245,"hour":4+6,"name":"F4"}
F3={"RA":352.3424	,"DEC":0.21245,"hour":2+6,"name":"F3"}
DT0={"RA":63.6937,"DEC":-7.6758,"hour":6+6,"name":"DT0"}

# Define the observer's location (Fort Sumner)
observer_location = EarthLocation(lat=34.169300*u.deg, lon=-105.533997*u.deg)

# Create an observer object for Fort Sumner
observer = Observer(location=observer_location, name="Fort Sumner")

date = '2023-09-17'
date = '2024-09-07'

start_time = Time(date + ' %02d:00:00' % (0))  # - 2*u.day #+ utcoffset
end_time = start_time + 25*u.day
times = time_grid_from_range((start_time, end_time), time_resolution=1*u.day)

df = pd.DataFrame(columns=["Target"] + [time.strftime("%m-%d") for time in times])
# Define the target coordinates (J2000)
for i,field in enumerate([TBS, BQSO2, F2, BQSO1, BQSO3, QSO1, QSO2,QSO3, F1,F3,F4, DT0]):
    start_time = Time(date + ' %02d:00:00' % (field["hour"]))  # - 2*u.day #+ utcoffset
    end_time = start_time + 25*u.day
    rai, deci, houri = field["RA"], field["DEC"], field["hour"]
    target_coordinates = SkyCoord(ra=rai*u.deg, dec=deci*u.deg)
    df.loc[i] = [field["name"]] + [int(get_moon(time, location=observer_location).separation(target_coordinates).to(u.deg).value) for time in times]
df.to_clipboard()



# get_moon(times[0], location=observer_location).separation(target_coordinates)

# get_moon(times[0], location=observer_location).separation_2d(target_coordinates)


#%%

import pandas as pd
from astropy.coordinates import EarthLocation, AltAz, get_moon, SkyCoord
from astropy.time import Time
import astropy.units as u
from astroplan import time_grid_from_range
from astroplan import Observer

# Define the observer's location (Fort Sumner)
observer_location = EarthLocation(lat=34.169300 * u.deg, lon=-105.533997 * u.deg)

# Create an observer object for Fort Sumner
observer = Observer(location=observer_location, name="Fort Sumner")


date = '2023-09-17'
date = '2024-09-07'

start_time = Time(date + ' %02d:00:00' % (0))  # - 2*u.day #+ utcoffset
end_time = start_time + 25 * u.day
times = time_grid_from_range((start_time, end_time), time_resolution=1 * u.day)

df = pd.DataFrame(columns=["Target"] + [time.strftime("%m-%d") for time in times])
# Define the target coordinates (J2000)
for i, field in enumerate([TBS, BQSO2, F2, BQSO1, BQSO3, QSO1, QSO2, QSO3, F1, F3, F4, DT0]):
    start_time = Time(date + ' %02d:00:00' % (field["hour"]))  # - 2*u.day #+ utcoffset
    end_time = start_time + 25 * u.day
    rai, deci, houri = field["RA"], field["DEC"], field["hour"]
    target_coordinates = SkyCoord(ra=rai * u.deg, dec=deci * u.deg, frame='icrs')
    
    # separation_altaz = []
    moons = [get_moon(time, location=observer_location).transform_to(AltAz(obstime=time, location=observer_location)) for time in times]
    targets = [target_coordinates.transform_to(AltAz(obstime=time, location=observer_location))   for time in times]
    # df.loc[3*i] = [field["name"]] + [int(get_moon(time, location=observer_location).separation(target_coordinates).to(u.deg).value) for time in times]


    df.loc[i] = [field["name"]+ " Az"] + [abs(int( moon_altaz.az.deg - target.az.deg  ))*np.cos( ((moon_altaz.alt.deg+target.alt.deg)/2)  *np.pi/180) if abs(int( moon_altaz.az.deg - target.az.deg  ))<180 else (360-abs(int( moon_altaz.az.deg - target.az.deg  )))*np.cos( ((moon_altaz.alt.deg+target.alt.deg)/2)  *np.pi/180) for time,target,moon_altaz in zip(times,targets,moons)]

    # df.loc[i] = [field["name"]+ " Az"] + [abs(int( moon_altaz.az.deg - target_coordinates.transform_to(AltAz(obstime=time, location=observer_location)).az.deg  ))*np.cos(moon_altaz.alt.deg*np.pi/180) if abs(int( moon_altaz.az.deg - target_coordinates.transform_to(AltAz(obstime=time, location=observer_location)).az.deg  ))<180 else (360-abs(int( moon_altaz.az.deg - target_coordinates.transform_to(AltAz(obstime=time, location=observer_location)).az.deg  )))*np.cos(moon_altaz.alt.deg*np.pi/180) for time,moon_altaz in zip(times,moons)]



    # df.loc[3*i+2] = [field["name"]+ " El"] + [abs(int( moon_altaz.alt.deg - target_coordinates.transform_to(AltAz(obstime=time, location=observer_location)).alt.deg)) for time,moon_altaz in zip(times,moons)]
    # df.loc[3*i+1] = [field["name"]+ " Az"] + [int( moon_altaz.az.deg - target_coordinates.transform_to(AltAz(obstime=time, location=observer_location)).az.deg  ) for time,moon_altaz in zip(times,moons)]
    # df.loc[3*i+1] = [field["name"]+ " El"] + [int( moon_altaz.alt.deg - target_coordinates.transform_to(AltAz(obstime=time, location=observer_location)).alt.deg) for time,moon_altaz in zip(times,moons)]

    # for time in times:
    #     moon_altaz = get_moon(time, location=observer_location).transform_to(AltAz(obstime=time, location=observer_location))
    #     target_altaz = target_coordinates.transform_to(AltAz(obstime=time, location=observer_location))
        
        
    #     df.loc[i] = [field["name"]] + [int(get_moon(time, location=observer_location).separation(target_coordinates).to(u.deg).value) for time in times]

        
        # separation_altaz_deg = moon_altaz.separation(target_altaz)
        # separation_altaz_elevation = separation_altaz_deg.alt.degree
        # separation_altaz_azimuth = separation_altaz_deg.az.degree
        # separation_altaz_str = f"Elevation: {separation_altaz_elevation:.2f}°, Azimuth: {separation_altaz_azimuth:.2f}°"
        # separation_altaz.append(separation_altaz_str)

    # df.loc[i] = [field["name"]] + separation_altaz

df.to_clipboard()



