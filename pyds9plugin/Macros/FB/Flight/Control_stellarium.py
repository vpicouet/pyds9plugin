import datetime
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun


def convert_to_julian(date_string="11/08/23 14:30:00", dformat="%d/%m/%y %H:%M:%S"):
    # Convert the input date string to a Python datetime object
    input_datetime = datetime.datetime.strptime(date_string, "%d/%m/%y %H:%M:%S")
    
    # Convert the datetime object to an Astropy Time object
    astropy_time = Time(input_datetime )- 0*u.hour
    
    # Get the Julian Date
    julian_date = astropy_time.jd
    return julian_date 


# meilleur direction de visée (soit uniquement IMU, soit DTU(avec offset), soit guider(avec la consigne en entrée ce qui n'est pas necessairement vrai) )
# RADEC: 
# 
# 
def follow_FB_LOS_on_Stellarium(path="/tmp/test.csv",ftype="replay",n_ra="DV Asc D",n_dec="DV Dec",gtype="radec",n_alt="DV EL",n_az="DV Az",lat="Latitude",lon="Longi",alt="Altitude",date="Datation GPS",dformat="%d/%m/%y %H:%M:%S"):
    
    i=0
    cat = Table.read(path,format="csv",delimiter=";")
    loc = EarthLocation(lat = cat[lat][-1]*u.deg, lon = cat[lon][-1]*u.deg, height = cat[alt][-1]*u.m)
    obj = SkyCoord(ra = cat[n_ra]*u.deg, dec = cat[n_dec]*u.deg)
    delta_midnight=1*u.min
    altaz = obj.transform_to(AltAz(obstime=Time([datetime.datetime.strptime(cat[date][i], dformat) for i in range(len(cat))])+delta_midnight, location = loc))
    cat["alt_from_ra"] =  np.array((altaz.alt)*u.deg)
    cat["az_from_ra"] =  np.array((altaz.az)*u.deg)

    n_az_b = n_az+"_b"
    cat[n_az_b] = 180 - cat[n_az]
    os.system('curl -d "latitude=%0.3f" http://localhost:8090/api/location/setlocationfields'%(cat[lat][0]))
    os.system('curl -d "longitude=%0.3f" http://localhost:8090/api/location/setlocationfields'%(cat[lon][0]))
    os.system('curl -d "altitude=%0.3f" http://localhost:8090/api/location/setlocationfields'%(cat[alt][0]))
    
    while 1>0:
        cat = Table.read(path,format="csv",delimiter=";")
        cat[n_az_b] =  180-cat[n_az]
        if gtype=="radec":
            a,b = n_ra, n_dec
        elif gtype=="altaz":
            b,a = n_alt, n_az_b

        
        if ftype=="flight":
            i=-1
        elif ftype=="replay":
            cat["x"] = np.cos(cat[b] * np.pi / 180) * np.cos(cat[a] * np.pi / 180)
            cat["y"] = np.cos(cat[b] * np.pi / 180) * np.sin(cat[a] * np.pi / 180)
            cat["z"] = np.sin(cat[b] * np.pi / 180)
            i+=1
            
        alpha, delta = cat[a][int(i)], cat[b][int(i)]
        x = np.cos(delta * np.pi / 180) * np.cos(alpha * np.pi / 180)
        y = np.cos(delta * np.pi / 180) * np.sin(alpha * np.pi / 180)
        z = np.sin(delta * np.pi / 180)
        
        if gtype=="radec":
            command = 'curl -d "j2000=[%0.10f,%0.10f,%0.10f]" http://localhost:8090/api/main/view'%(x,y,z)
            os.system('curl -d "time=%0.10f" http://localhost:8090/api/main/time'%(convert_to_julian(cat[date][i], dformat)))
            os.system(command)
        elif gtype=="altaz":
            command = ['curl -d "altAz=[%0.10f,%0.10f,%0.10f]" http://localhost:8090/api/main/view'%(x,y,z)]
            command.append('curl -d "time=%0.10f" http://localhost:8090/api/main/time'%(convert_to_julian(cat[date][i], dformat)))
            # os.system('curl -d "latitude=%0.3f" http://localhost:8090/api/location/setlocationfields'%(cat[lat][i]))
            # os.system('curl -d "longitude=%0.3f" http://localhost:8090/api/location/setlocationfields'%(cat[lon][i]))
            # os.system('curl -d "altitude=%0.3f" http://localhost:8090/api/location/setlocationfields'%(cat[alt][i]))
            os.system(";".join(command))
        if 1==0:
            # loc = EarthLocation(lat = cat[lat][i]*u.deg, lon = cat[lon][i]*u.deg, height = cat[alt][i]*u.m)
            # obj = SkyCoord(ra = cat[n_ra][i]*u.deg, dec = cat[n_dec][i]*u.deg)
            # delta_midnight=0
            # altaz = obj.transform_to(AltAz(obstime=Time(datetime.datetime.strptime(cat[date][i], dformat))+delta_midnight, location = loc))
            # print("Az_d, Alt_d = %0.3f, %0.3f"%(altaz.az.deg,altaz.alt.deg))
            print("Az, Alt = %0.3f, %0.3f"%(cat[n_az][i],cat[n_alt][i]))
            print("RA, DEC = %0.3f, %0.3f"%(cat[n_ra][i],cat[n_dec][i]))
        # print(command)
        time.sleep(0.1)
        # sys.exit()


follow_FB_LOS_on_Stellarium(path = "/Users/Vincent/Github/FB-dashboard/Catalogs/CNES/ScienceDataFile_skytest_Caltech.txt",ftype="replay",gtype="radec")
# follow_FB_LOS_on_Stellarium(path = "/Users/Vincent/Github/FB-dashboard/Catalogs/CNES/ScienceDataFile_skytest_Caltech.txt",ftype="replay",gtype="altaz")


#%%
alt,az = 0,180-90
print(np.cos(az * np.pi / 180) * np.cos(alt * np.pi / 180),np.cos(az * np.pi / 180) * np.sin(alt * np.pi / 180),np.sin(az * np.pi / 180))


#%%

i=0
cat = Table.read(path,format="csv",delimiter=";")
loc = EarthLocation(lat = cat[lat][-1]*u.deg, lon = cat[lon][-1]*u.deg, height = cat[alt][-1]*u.m)
obj = SkyCoord(ra = cat[n_ra]*u.deg, dec = cat[n_dec]*u.deg)
delta_midnight=1*u.min#0.025 *u.hour
altaz = obj.transform_to(AltAz(obstime=Time([datetime.datetime.strptime(cat[date][i], dformat) for i in range(len(cat))])+delta_midnight, location = loc))
cat["alt_from_ra"] =  np.array((altaz.alt)*u.deg)
cat["az_from_ra"] =  np.array((altaz.az)*u.deg)

fig, (ax0,ax1) = plt.subplots(2,1, sharex=True)
# ax0.plot(cat["x"],":.");ax1.plot(cat["y"],":.")
# ax0.plot(3600*cat[n_ra],":.");ax1.plot(3600*cat[n_dec],":.")
ax0.plot(cat["DV Az"],":.");ax1.plot(cat["DV EL"],":.")
# ax0.plot(cat["az_from_ra"],":.");ax1.plot(cat["alt_from_ra"],":.")
# n=50
# ax0.plot(3600*(cat["DV Az"][n:]-cat["az_from_ra"][n:]),":.");ax1.plot(3600*(cat["DV EL"][n:]-cat["alt_from_ra"][n:]),":.")
# ax0.set_title(np.ptp(3600*(cat["DV Az"][n:]-cat["az_from_ra"][n:])),)
# ax1.set_title(np.ptp(3600*(cat["DV EL"][n:]-cat["alt_from_ra"][n:])),)
# plt.plot(3600*cat[n_ra],3600*cat[n_dec],":.")

