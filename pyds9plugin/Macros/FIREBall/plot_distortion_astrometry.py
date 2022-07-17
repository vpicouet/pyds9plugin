
#%%
from astropy.io import fits
from astropy import wcs, coordinates
from astropy.table import Table
# import numpy as np
# import matplotlib.pyplot as plt
# filename = "/Volumes/ExtremePro/LAM/FIREBALL/2022/GuiderData/astrometry/19_bis.fits"
shadow = 500
WCS = wcs.WCS(filename)
header = fits.getheader(filename)
fits = fits.open(filename)

corners = np.array([[1, 1], [header["NAXIS1"],header["NAXIS2"]]])
corner_angles = WCS.all_pix2world(corners, 1)

field = corner_angles[1,:]-corner_angles[0,:]
cdelt = field/np.array([header["NAXIS1"],header["NAXIS2"]])


# distortion map
x = np.linspace(1,header["NAXIS1"],21)
y = np.linspace(1,header["NAXIS2"],21)
xx,yy = np.meshgrid(x, y)

xxd, yyd = WCS.sip_pix2foc(xx, yy, 1)
xxd -=  - WCS.wcs.crpix[0]
yyd -=  - WCS.wcs.crpix[1]

fig, (ax0,ax1) = plt.subplots(2,1,sharex=True, sharey=True,figsize=(8,8))
mask = xx>shadow
mean_x = np.mean(abs(xxd[mask]-xx[[mask]]))
mean_y = np.mean(abs(yyd[mask]-yy[[mask]]))
max_x = np.max(abs(xxd[mask]-xx[[mask]]))
max_y = np.max(abs(yyd[mask]-yy[[mask]]))
# print(yyd[mask]-yy[mask])
label =  "mean_x = %0.1f pix\nmean_y = %0.1f pix\nmax_x = %0.1f pix\nmax_y = %0.1f pix"%(mean_x,mean_y,max_x,max_y)
qv = ax0.quiver(xx, yy, xxd-xx, yyd-yy, scale=1 ,label=label, scale_units='xy', angles='xy' )            
ax0.quiverkey(qv, .2, -.1, 10, "10 pix")
ax0.set_xlabel("X")
ax0.set_ylabel("Y")
# ax0.set_xlim([500,header["NAXIS1"]])
# ax0.set_xlim((400,1000))
# ax0.set_ylim([header["NAXIS2"],0])
ax0.fill_between([0,shadow],[header["NAXIS2"],header["NAXIS2"]],color="k",alpha=0.2,label="Shadow")
if "ROTENC" in list(dict.fromkeys(header.keys())):
    ax0.set_title("%s - PA = %0.1f"%(os.path.basename(filename),header["ROTENC"]))
else:
    ax0.set_title("%s"%(os.path.basename(filename)))
# try:
#     for i in range(8):
#       if header["SIGMAX%i"%(i)]>0:
#         ax0.plot(header["CX%i"%(i)],header["CY%i"%(i)],"or",ms=10)
# except Exception as e:
#     print(e)

if os.path.isfile(filename.replace(".fits","_cat.fits")):
    cat = Table.read(filename.replace(".fits","_cat.fits"))
    ax0.scatter(cat["index_x"],cat["index_y"],marker="x",c = cat["match_weight"],label="Observed star")
    ax0.scatter(cat["field_x"],cat["field_y"],marker="+",c = cat["match_weight"],label="No distortion star position")
    ax1.scatter(cat["field_x"],cat["field_y"],marker="+",c = cat["match_weight"])
    ax1.scatter(cat["index_x"],cat["index_y"],marker="x",c = cat["match_weight"])

    ax1.quiver(cat["index_x"],cat["index_y"],cat["field_x"]-cat["index_x"],cat["field_y"]-cat["index_y"],color="r", scale_units='xy', angles='xy',scale=1 )   
    ax0.quiver(cat["index_x"],cat["index_y"],cat["field_x"]-cat["index_x"],cat["field_y"]-cat["index_y"],color="r", scale_units='xy', angles='xy' ,scale=1)   
    
ax0.legend()


for i in range(x.size):
    ax1.plot(xx[i,:], yy[i,:],':g' )
    ax1.plot(xx[:,i], yy[:,i], ':g' )
    ax1.plot(xxd[i,:], yyd[i,:],'b' )
    ax1.plot(xxd[:,i], yyd[:,i], 'b' )
if "ROTENC" in list(dict.fromkeys(header.keys())):
    ax1.set_title("PA = %0.1f - %s"%(header["ROTENC"],header["COMMENT"][-12]))
else:
    ax1.set_title("%s"%(os.path.basename(filename)))
ax1.set_xlabel("X")
ax1.fill_between([0,shadow],[header["NAXIS2"],header["NAXIS2"]],color="k",alpha=0.2)
# try:
#     for i in range(8):
#       if header["SIGMAX%i"%(i)]>0:
#           ax1.plot(header["CX%i"%(i)],header["CY%i"%(i)],"or",ms=10)
# except Exception as e:
#     print(e)

fig.tight_layout()
fig.savefig(filename.replace(".fits",".png"))
plt.show()

# %%
