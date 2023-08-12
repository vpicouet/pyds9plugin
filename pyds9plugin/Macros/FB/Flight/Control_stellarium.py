import datetime
import os
import time
from astropy import units as u
from astropy.table import Table
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import sys
import threading

# I should remove points that are different by > 30 " de chauqe coté

def convert_to_julian(date_string="11/08/23 14:30:00", dformat="%d/%m/%y %H:%M:%S"):
    # Convert the input date string to a Python datetime object
    input_datetime = datetime.datetime.strptime(date_string, dformat)
    
    # Convert the datetime object to an Astropy Time object
    astropy_time = Time(input_datetime )- 0*u.hour
    
    # Get the Julian Date
    julian_date = astropy_time.jd
    return julian_date 


# meilleur direction de visée (soit uniquement IMU, soit DTU(avec offset), soit guider(avec la consigne en entrée ce qui n'est pas necessairement vrai) )
# RADEC: 
# 
# 
def follow_FB_LOS_on_Stellarium(path="/tmp/test.csv",ftype="replay",guidance="Best",n_ra="DV Asc D",n_dec="DV Dec",gtype="radec",n_alt="DV EL",n_az="DV Az",lat="Latitude",lon="Longi",alt="Altitude",date="Datation GPS",dformat="%d/%m/%y %H:%M:%S",sleep=0.1,delay=1*u.min):
    print("path = ",path,"ftype=",ftype,"gtype=",gtype,"sleep=",sleep,"guidance=",guidance)

    i=0
    cat = Table.read(path,format="csv")
    GPS = np.array([np.ma.median(cat[lat]),  np.ma.median(cat[lon]), np.ma.median(cat[alt])])
    GPS[~np.isfinite(GPS)] = 1
    loc = EarthLocation(lat = GPS[0]*u.deg, lon =GPS[0]*u.deg, height = GPS[0]*u.m)
    obj = SkyCoord(ra = cat[n_ra]*u.deg, dec = cat[n_dec]*u.deg)
    
    altaz = obj.transform_to(AltAz(obstime=Time([datetime.datetime.strptime(cat[date][i], dformat) for i in range(len(cat))])+delay, location = loc))
    cat["alt_from_ra"] =  np.array((altaz.alt)*u.deg)
    cat["az_from_ra"] =  np.array((altaz.az)*u.deg)

    n_az_b = n_az+"_b"
    cat[n_az_b] = 180 - cat[n_az]
    os.system('curl -d "latitude=%0.3f" http://localhost:8090/api/location/setlocationfields > /dev/null 2>&1'%(GPS[0]))
    os.system('curl -d "longitude=%0.3f" http://localhost:8090/api/location/setlocationfields  > /dev/null 2>&1'%(GPS[1]))
    os.system('curl -d "altitude=%0.3f" http://localhost:8090/api/location/setlocationfields  > /dev/null 2>&1'%(GPS[2]))
    
    while 1>0:
        cat = Table.read(path,format="csv")
        # cat = cat[cat["Mins_after_launch"]>500]
        while len(cat.colnames)<5:
            cat = Table.read(path,format="csv")
            
        cat[n_az_b] =  180-cat[n_az]
        if "radec" in gtype:
            a,b = n_ra, n_dec
        elif gtype=="altaz":
            b,a = n_alt, n_az_b

        
        if ftype=="real-time":
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
            command = ['curl -d "j2000=[%0.10f,%0.10f,%0.10f]" http://localhost:8090/api/main/view  > /dev/null 2>&1'%(x,y,z)]
            command.append(' -d "time=%0.10f" http://localhost:8090/api/main/time  > /dev/null 2>&1'%(convert_to_julian(cat[date][i], dformat)))
            os.system(" --next ".join(command))
        elif gtype=="radec-notime":
            print(alpha, delta)
            # print(x,y,z)
            command = 'curl -d "j2000=[%0.10f,%0.10f,%0.10f]" http://localhost:8090/api/main/view  > /dev/null 2>&1'%(x,y,z)
            os.system(command)
        elif gtype=="radec-lowtime":
            if i%50==0:
                command = ['curl -d "j2000=[%0.10f,%0.10f,%0.10f]" http://localhost:8090/api/main/view  > /dev/null 2>&1'%(x,y,z)]
                command.append(' -d "time=%0.10f" http://localhost:8090/api/main/time  > /dev/null 2>&1'%(convert_to_julian(cat[date][i], dformat)))
                os.system(" --next ".join(command))
            else:
                command = 'curl -d "j2000=[%0.10f,%0.10f,%0.10f]" http://localhost:8090/api/main/view  > /dev/null 2>&1'%(x,y,z)
                os.system(command)

        elif gtype=="altaz":
            command = ['curl -d "altAz=[%0.10f,%0.10f,%0.10f]" http://localhost:8090/api/main/view  > /dev/null 2>&1'%(x,y,z)]
            command.append(' -d "time=%0.10f" http://localhost:8090/api/main/time  > /dev/null 2>&1'%(convert_to_julian(cat[date][i], dformat)))
            # command.append(' -d "latitude=%0.3f" http://localhost:8090/api/location/setlocationfields  > /dev/null 2>&1'%(cat[lat][i]))
            # command.append(' -d "longitude=%0.3f" http://localhost:8090/api/location/setlocationfields  > /dev/null 2>&1'%(cat[lon][i]))
            # command.append(' -d "altitude=%0.3f" http://localhost:8090/api/location/setlocationfields  > /dev/null 2>&1'%(cat[alt][i]))
            os.system(" --next ".join(command))
        if 1==0:
            # loc = EarthLocation(lat = cat[lat][i]*u.deg, lon = cat[lon][i]*u.deg, height = cat[alt][i]*u.m)
            # obj = SkyCoord(ra = cat[n_ra][i]*u.deg, dec = cat[n_dec][i]*u.deg)
            # delta_midnight=0
            # altaz = obj.transform_to(AltAz(obstime=Time(datetime.datetime.strptime(cat[date][i], dformat))+delta_midnight, location = loc))
            # print("Az_d, Alt_d = %0.3f, %0.3f"%(altaz.az.deg,altaz.alt.deg))
            print("Az, Alt = %0.3f, %0.3f"%(cat[n_az][i],cat[n_alt][i]))
            print("RA, DEC = %0.3f, %0.3f"%(cat[n_ra][i],cat[n_dec][i]))
        # print(command)
        time.sleep(sleep)
        # sys.exit()
 


class TestThreading(object):
    def __init__(self, interval=1,catalog="",saveto="/tmp/test.csv"):
        self.interval = interval
        self.catalog = catalog
        self.saveto = saveto

        thread = threading.Thread(target=self.write_csv, args=())
        thread.daemon = True
        thread.start()

    def write_csv(self):
        cat = Table.read(self.catalog)
        i=20
        cat[:i].write(self.saveto,overwrite=True)
        while True:
            cat[:i+1].write(self.saveto,overwrite=True)
            i+=1
            time.sleep(self.interval)



if sys.argv[-3] == "/Users/Vincent/Github/FB-dashboard/Catalogs/CNES/FlightPrincipalCatalog_v5_tronc.csv":
    n_ra="BRD_VM_30MS_1_TM_DEBUG1_AscDvEstGuider"
    n_dec="BRD_VM_30MS_1_TM_DEBUG1_DecDvEstGuider"
    n_alt="DV_EL"
    n_az="DV_Az"
    alt="altitudes[Feets]"
    date="Datation_GPS"
    dformat="%Y-%m-%dT%H:%M:%S"
else:
    n_ra="DV Asc"
    n_dec="DV Dec"
    gtype="radec"
    n_alt="DV EL"
    n_az="DV Az"
    alt="Altitude"
    date="Datation GPS"
    dformat="%d/%m/%y %H:%M:%S"


# follow_FB_LOS_on_Stellarium(path = sys.argv[-3],ftype=sys.argv[-2],gtype=sys.argv[-1],sleep=float(sys.argv[-4]),guidance=sys.argv[-5])
if sys.argv[-2]=="real-time-test":
    tr = TestThreading(catalog =sys.argv[-3],saveto="/tmp/test.csv" )
    time.sleep(1)
    sys.argv[-3] = "/tmp/test.csv"

    # follow_FB_LOS_on_Stellarium(path = "/tmp/test.csv",ftype=sys.argv[-2],gtype=sys.argv[-1],sleep=float(sys.argv[-4]),guidance=sys.argv[-5],n_ra="BRD_VM_30MS_1_TM_DEBUG1_AscDvEstGuider",n_dec="BRD_VM_30MS_1_TM_DEBUG1_DecDvEstGuider",n_alt="DV_EL",n_az="DV_Az",lat="Latitude",lon="Longi",alt="altitudes[Feets]",date="Datation_GPS",dformat="%Y-%m-%dT%H:%M:%S")
    

    
follow_FB_LOS_on_Stellarium(path = sys.argv[-3],ftype=sys.argv[-2],gtype=sys.argv[-1],sleep=float(sys.argv[-4]),guidance=sys.argv[-5],n_ra=n_ra,n_dec=n_dec,n_alt=n_alt,n_az=n_az,date=date,dformat=dformat)
    
    
    
    

    
    # follow_FB_LOS_on_Stellarium(path = "/Users/Vincent/Github/FB-dashboard/Catalogs/CNES/FlightPrincipalCatalog_v5_tronc.csv",gtype="radec-notime",ftype="replay",n_ra="BRD_VM_30MS_1_TM_DEBUG1_AscDvEstGuider",n_dec="BRD_VM_30MS_1_TM_DEBUG1_DecDvEstGuider",n_alt="DV_EL",n_az="DV_Az",lat="Latitude",lon="Longi",alt="altitudes[Feets]",date="Datation_GPS",dformat="%Y-%m-%dT%H:%M:%S",sleep=0.1)
# ,altitudes[Feets],ROTENC,,RA_DTU,DEC_DTU,ROLL_DTU,DV_Az,DV_EL,MROT,Moon_El,Moon_Az,Sun_El,Sun_Az,F2_El,F2_Az,m31_El,m31_Az,Latitude,Longi,Altitude,,BRD_VM_30MS_1_TM_DEBUG1_AscDvEstDtu,BRD_VM_30MS_1_TM_DEBUG1_AscDvEstGuider,DV_Asc_D,BRD_VM_30MS_1_TM_DEBUG1_DecDvEstDtu,BRD_VM_30MS_1_TM_DEBUG1_DecDvEstGuider,DV_Dec,Rot_Angle,

# follow_FB_LOS_on_Stellarium(path = "/Users/Vincent/Github/FB-dashboard/Catalogs/CNES/ScienceDataFile_skytest_Caltech.txt",ftype="replay",gtype="altaz")


#%%
# alt,az = 0,180-90
# print(np.cos(az * np.pi / 180) * np.cos(alt * np.pi / 180),np.cos(az * np.pi / 180) * np.sin(alt * np.pi / 180),np.sin(az * np.pi / 180))
# path = "/Users/Vincent/Github/FB-dashboard/Catalogs/CNES/FlightPrincipalCatalog_v5_tronc.csv";ftype="replay";gtype="radec";n_ra="DV_Asc_D";n_dec="DV_Dec";n_alt="DV EL";n_az="DV Az";lat="Latitude";lon="Longi";alt="altitudes[Feets]";date="Datation_GPS";dformat="%d/%m/%y %H:%M:%S";sleep=0.1

# #%%

# i=0
# cat = Table.read(path,format="csv",delimiter=";")
# loc = EarthLocation(lat = cat[lat][-1]*u.deg, lon = cat[lon][-1]*u.deg, height = cat[alt][-1]*u.m)
# obj = SkyCoord(ra = cat[n_ra]*u.deg, dec = cat[n_dec]*u.deg)
# delta_midnight=1*u.min#0.025 *u.hour
# altaz = obj.transform_to(AltAz(obstime=Time([datetime.datetime.strptime(cat[date][i], dformat) for i in range(len(cat))])+delta_midnight, location = loc))
# cat["alt_from_ra"] =  np.array((altaz.alt)*u.deg)
# cat["az_from_ra"] =  np.array((altaz.az)*u.deg)

# fig, (ax0,ax1,ax2,ax3) = plt.subplots(4,1, sharex=True)
# # ax0.plot(cat["x"],":.");ax1.plot(cat["y"],":.")
# ax0.plot(3600*(cat[n_ra][1:]-cat[n_ra][:-1]),":.");ax1.plot(3600*(cat[n_dec][1:]-cat[n_dec][:-1]),":.")
# ax2.plot(3600*cat[n_ra],":.");ax3.plot(3600*cat[n_dec],":.")
# ax0.plot(cat["DV Az"],":.");ax1.plot(cat["DV EL"],":.")
# ax0.plot(cat["az_from_ra"],":.");ax1.plot(cat["alt_from_ra"],":.")
# n=50
# ax0.plot(3600*(cat["DV Az"][n:]-cat["az_from_ra"][n:]),":.");ax1.plot(3600*(cat["DV EL"][n:]-cat["alt_from_ra"][n:]),":.")
# ax0.set_title(np.ptp(3600*(cat["DV Az"][n:]-cat["az_from_ra"][n:])),)
# ax1.set_title(np.ptp(3600*(cat["DV EL"][n:]-cat["alt_from_ra"][n:])),)
# plt.plot(3600*cat[n_ra],3600*cat[n_dec],":.")

