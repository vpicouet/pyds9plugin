
# from astropy.table import hstack
paths = get(d,"Path of the images you want to stack:")
print(paths)
if paths!="":
    files=globglob(paths)
else:
    files=get_filename(d, All=True, sort=False)
files.sort()
region = getregion(d, quick=True, message=False, selected=True)
print("region = ", region)
Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
new_image = np.hstack([fits.open(f)[0].data[Yinf:Ysup, Xinf:Xsup] for f in files]) 
print(new_image)
# fitsimage.data = new_image
# fitsimage.write("/tmp/test.fits")
name = files[0].replace(".fits","_TF.fits")
fitswrite(new_image,name)
d.set("frame new; file "+name)

param_dict = {"DETECT_THRESH":,10
"GAIN":,0
"DETECT_MINAREA":10,
"DEBLEND_NTHRESH":64,
"DEBLEND_MINCONT":0.3,
"CLEAN":,0
"CLEAN_PARAM":,0
}

run_sep(name, name, param_dict)

a = Table.read(name.replace(".fits","_cat.fits"))
import matplotlib.pyplot as plt
import numpy as np

# créer des données pour les tracés
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x)
y3 = np.cos(x)
y4 = np.log(x)

# créer la figure et les sous-graphes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

# définir les couleurs des axes y
color1 = 'tab:red'
color2 = 'tab:blue'

# tracer les courbes sur chaque sous-graphe
ax1.plot(x, y1, color=color1)
ax1.set_ylabel('sin(x)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title('Subplot 1')
ax1b = ax1.twinx()
ax1b.plot(x, y2, color=color2)
ax1b.set_ylabel('exp(x)', color=color2)
ax1b.tick_params(axis='y', labelcolor=color2)

ax2.plot(x, y2, color=color2)
ax2.set_ylabel('exp(x)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_title('Subplot 2')
ax2b = ax2.twinx()
ax2b.plot(x, y1, color=color1)
ax2b.set_ylabel('sin(x)', color=color1)
ax2b.tick_params(axis='y', labelcolor=color1)

ax3.plot(x, y3, color=color1)
ax3.set_ylabel('cos(x)', color=color1)
ax3.tick_params(axis='y', labelcolor=color1)
ax3.set_title('Subplot 3')
ax3b = ax3.twinx()
ax3b.plot(x, y4, color=color2)
ax3b.set_ylabel('log(x)', color=color2)
ax3b.tick_params(axis='y', labelcolor=color2)

ax4.plot(x, y4, color=color2)
ax4.set_ylabel('log(x)', color=color2)
ax4.tick_params(axis='y', labelcolor=color2)
ax4.set_title('Subplot 4')
ax4b = ax4.twinx()
ax4b.plot(x, y3, color=color1)
ax4b.set_ylabel('cos(x)', color=color1)
ax4b.tick_params(axis='y', labelcolor=color1)

# ajuster la disposition des sous-graphes
fig.tight_layout()

# afficher la figure
plt.show()
