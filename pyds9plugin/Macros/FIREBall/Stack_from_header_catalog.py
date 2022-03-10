from pyds9plugin.DS9Utils import *



path = "/Users/Vincent/Nextcloud/LAM/FIREBALL/2019-01+MarseilleTestsImages/DetectorAnalysis/TestVincent/HeaderCatalogbasic_image_estimators.csv"#get(d, "Path of the header catalog :", exit_=True)
path = "/Users/Vincent/Nextcloud/LAM/FIREBALL/2019-01+MarseilleTestsImages/DetectorAnalysis/TestVincent/190206/highsignal/HeaderCatalogbasic_image_estimators.csv"#get(d, "Path of the header catalog :", exit_=True)

from astropy.table import Table
try:
    cat = Table.read(path)
except:
    verboseprint("Impossible to open table, verify your path.")

#################FIELD YOU WANT PRESERVE
path_field = "Path"
Field1 = "dir_path"
Field2 = "EMGAIN"#my_conf.gain[0]
Field3 = "EXPTIME"#my_conf.exptime[0]
Field4 = "FileSize_Mo"#"EMCCDBAC"


param_list =[
            {'name':"dir_path", 'round':None},
            {'name':"EMGAIN", 'round':None},
            {'name':"EXPTIME", 'round':None},
            {'name':"FileSize_Mo", 'round':1}
]

cat_pd = cat.to_pandas()
for name in param_list:
    print(cat_pd[name['name']])

# n=len(param_list)

# for i in range(len(param_list)):
#     if param_list[i]['round'] is None:
#         param_list[i]['list'] = np.unique(cat[param_list[i]['name']])
#     else:
#         param_list[i]['list'] = np.unique(np.round(cat[param_list[i]['name']],param_list[i]['round']))

# # cat_pd = cat.to_pandas()
# for name in 


#do a recursive loop

verboseprint(cat)
verboseprint(cat[Field1])
fpath = np.unique(cat[Field1])
if type(fpath) == str:
    fpath = [fpath]
for path in fpath:
    verboseprint(path)
    for gain in np.unique(cat[(cat[Field1] == path)][Field2]):
        for exp in np.unique(cat[(cat[Field1] == path) & (cat[Field2] == gain)][Field3]):
            for temp in np.unique(cat[(cat[Field1] == path) & (cat[Field2] == gain) & (cat[Field3] == exp)][Field4].astype(float)).astype(int):
                files = cat[([path in cati[Field1] for cati in cat]) & (cat[Field2] == gain) & (cat[Field3] == exp) & (cat[Field4].astype(float).astype(int) == temp)][path_field]
                # print(files)
                if len(files) > 0:
                    verboseprint("Stacking images of GAIN = %i, exposure  = %i, temp = %i and path=%s" % (float(gain), float(exp), float(temp), path))
                    verboseprint(files)
                    stack_images_path(files, fname="-Gain%i-Texp%i-Temp%i" % (float(gain), float(exp), float(temp)),Type="nanmedian",)
