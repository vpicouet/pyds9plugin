from pyds9plugin.DS9Utils import DS9n, get_filename, get, yesno, globglob, verboseprint
from astropy.io import fits
import datetime
import numpy as np
import time
import os

d=DS9n()
filename = get_filename(d)
from astropy.table import Table
plot_=True

def AddTimeSec(table, TUtimename="time", NewTime_s="DATE_s", timeformat="%m/%d/%y %H:%M:%S"):
    import time

    # sec = [datetime.datetime.strptime(moment, timeformat) for moment in table[TUtimename]]
    sec = [time.mktime(datetime.datetime.strptime(moment, timeformat).timetuple()) for moment in table[TUtimename]]
    table[NewTime_s] = np.array(sec)
    return table


def FindTimeField(liste):
    timestamps = ["Datation GPS", "Date", "Time", "Date_TU", "UT Time", "Date_Time", "Time UT","DATETIME"]
    timestamps_final = [field.upper() for field in timestamps] + [field.lower() for field in timestamps] + timestamps
    try:
        timestamps_final.remove("date")
    except ValueError:
        pass
    for timename in timestamps_final:
        if timename in liste:  # table.colnames:
            timefield = timename
    try:
        verboseprint("Time field found : ", timefield)
        return timefield
    except UnboundLocalError:
        return "DATETIME"#liste[0]  # table.colnames[0]


def RetrieveTimeFormat(time):
    formats = ["%d/%m/%Y %H:%M:%S.%f", '%Y-%m-%d %H:%M:%S.%f',"%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S"]
    form = []
    for formati in formats:
        try:
            datetime.datetime.strptime(time, formati)
            form.append(True)
        except ValueError:
            form.append(False)
    return formats[np.argmax(form)]


# catalog, Field, timediff, timeformatImage, timeformatCat, filename = "/Users/Vincent/Downloads/Safari/alltemps-4.csv",  "time", 7 ,'%Y-%m-%d %H:%M:%S.%f', "%m/%d/%y %H:%M:%S",  "/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/20220430/image_00000*.fits"

catalog = get(d, 'Catalog to interpolate data from', exit_=True)
timediff = int(get(d, 'Time difference', exit_=True))
if yesno(d, "Do you want to run this on all the folder images"):
    filename = os.path.dirname(filename) + '/image_*.fits'

timeformatCat, timeformatImage = "-", "-"
# print("Field, catalog, timediff, formatImage, formatCat =", Field, catalog, timediff, timeformatImage, timeformatCat)
# if len(sys.argv) > 3+5: path = Charge_path_new(filename, entry_point=3+5)
# path = Charge_path_new(filename) if len(sys.argv) > 8 else [filename]  # and verboseprint('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
path = globglob(filename)
header0 = fits.getheader(path[0])
try:
    header0['DATETIME'] = header0['OBSDATE'] + " " +header0['OBSTIME']
except KeyError:
    pass
TimeFieldImage = FindTimeField(list(set(header0.keys())))#'DATETIME'#
if (timeformatImage == "-") or (timeformatImage == "'-'"):
    timeformatImage = RetrieveTimeFormat(header0[TimeFieldImage])
timestamp_image = datetime.datetime.strptime(header0[TimeFieldImage], timeformatImage)
# Field = 'EMCCDBack[C]'
cat = Table.read(catalog)
TimeFieldCat = FindTimeField(cat.colnames)
if (timeformatCat == "-") or (timeformatCat == "'-'"):
    timeformatCat = RetrieveTimeFormat(cat[TimeFieldCat][0])
cat['timestamp'] =[ datetime.datetime.strptime(d, timeformatCat) for d in cat[TimeFieldCat]]#.timestamp()


cat = AddTimeSec(cat, TUtimename=TimeFieldCat, NewTime_s="DATE_s", timeformat=timeformatCat)
cat.sort("DATE_s")
# verboseprint(type(cat[Field][0]), type(cat["DATE_s"][0]))
ok = []
# try:
#     # yy = np.array(cat[Field], dtype=float)
#     yy = np.array(cat["DATE_s"], dtype=float)
#     t = cat["DATE_s"]
# except ValueError:
#     ok = [a.replace("-", ".").replace(".", "0").isdigit() for a in cat[Field]]
#     verboseprint(ok)
#     t, yy = cat["DATE_s"][ok], np.array(cat[Field][ok], dtype=float)
# verboseprint(t, yy)
# plt.plot(t, yy, linestyle="dotted", label="Catalog Field")
# plt.plot([datetime.datetime.strptime(d, '%m/%d/%y %H:%M:%S') for d in cat[TimeFieldCat]],yy, linestyle="dotted", label="Catalog Field");plt.gcf().autofmt_xdate()
if plot_:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([datetime.datetime.strptime(d, '%m/%d/%y %H:%M:%S') for d in cat[TimeFieldCat]],cat[cat.colnames[1]], linestyle="dotted", label="Catalog Field");plt.gcf().autofmt_xdate()
# t, y = t[conv_kernel:-conv_kernel], np.convolve(yy, np.ones(conv_kernel) / conv_kernel, mode="same")[conv_kernel:-conv_kernel]
# # plt.plot(t, y, label="Field convolved")

# t, index = np.unique(t, return_index=True)
# y = y[index]
# FieldInterp = interpolate.interp1d(t, np.array(y), kind="linear")




for filename in path:
    verboseprint(filename)
    header = fits.getheader(filename)
    header['DATETIME'] = header['OBSDATE'] + " " +header['OBSTIME']

    timestamp_image = datetime.datetime.strptime(header[TimeFieldImage], timeformatImage)
    timeseconds = time.mktime(datetime.datetime.strptime(header[TimeFieldImage], timeformatImage).timetuple())
    datetime.datetime.strptime(header[TimeFieldImage], timeformatImage).timetuple()
    # try:
    #     value = FieldInterp(timeseconds - float(timediff) * 3600)
    #     # plt.plot(timeseconds, value, "o", c="red")  # label='Images after change time')
    #     # plt.plot(datetime.datetime.strptime(header[TimeFieldImage], timeformatImage), value, ".", c="red")
    #     # plt.plot(datetime.datetime.strptime(header[TimeFieldImage], timeformatImage)- datetime.timedelta(hours=timediff), value, ".", c="green")
    #     # label='Images after change time')
    # except ValueError:
    #     value = -999
    #     # plt.plot(datetime.datetime.strptime(header[TimeFieldImage], timeformatImage), np.nanmean(yy), "P", c="black")  # , label='Images out of interpoalation range: -999')
    # value = np.round(value, 12)




    if "NAXIS3" in header:
        fits.delval(filename, "NAXIS3")
        verboseprint("2D array: Removing NAXIS3 from header...")
    # fits.setval(filename, Field[:8], value=value)#, comment=comment)
    columns = cat.colnames[1:]
    columns.remove('timestamp')
    for i, column in enumerate(columns):
        mask = np.isfinite(cat[column])
        temp = cat[column][mask][np.argmin(abs(cat[mask]['timestamp']+ datetime.timedelta(hours=timediff)-timestamp_image))]
        fits.setval(filename, column[:8], value=temp)#, comment=comment)
        if plot_:
            if i==0:
                plt.plot(datetime.datetime.strptime(header[TimeFieldImage], timeformatImage)- datetime.timedelta(hours=timediff), temp, ".", c="green")
    #        verboseprint(value)
    #        fitswrite(fitsimage,filename)
if plot_:
    plt.plot([], [], "P", c="black", label="Images out of interpoalation range: -999")
    plt.plot([], [], "o", c="red", label="Images after change time")
    # plt.plot(np.linspace(t.min(),t.max(),1e5), FieldInterp(np.linspace(t.min(),t.max(),1e5)), c = 'grey', label='Linear interplolation')
    plt.title("Addition of the catlog field to image headers")
    plt.xlabel("Time in seconds")
    plt.ylabel(columns[0])
    plt.legend()
    plt.show()
