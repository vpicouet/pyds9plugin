from astropy.table import Table
fields = get(d, "What field do you want to plot? Possibilities = F1,F2,F3,F4,QSO1,QSO2,QSO3,QSO4,BQSO1,BQSO2,BQSO3").rstrip().split(",")

# print(fields)
def DS9plot_field(field):    
    if field == "F1":
        ra,dec, angle = 32.19,	-5.688,	0
    if field == "F2":
        ra,dec, angle = 253.0624,	34.9699,	20
    if field == "F3":
        ra,dec, angle =352.3424,	0.21245,	0
    if field == "F4":
        ra,dec, angle = 36.9049,	0.65245,	0
    if field == "QSO1":
        ra,dec, angle = 0.03894,	1.39459,	10
    if field == "QSO2":
        ra,dec, angle = 22.42638,	0.62279,	-20
    if field == "QSO3":
        ra,dec, angle = 14.62883,	0.00512,	10
    if field == "QSO4":
        ra,dec, angle = 351.8533,	-1.8553,	0
    
    if field == "BQSO1":
        ra,dec, angle = 275.48,	64.36,	-70
    if field == "BQSO2":
        ra,dec, angle = 234.89,	47.59,	120
    if field == "BQSO3":
        ra,dec, angle = 11.865604,	3.244401,	0
    
    d.set("frame new")
    # d.set("skyview size 1.5 1.5 degrees;skyview pixels 600 600 ; skyview %0.3f %0.3f; "%(ra,dec))
    # d.set("dsssao size 1.5 1.5 degrees;dsssao %0.3f %0.3f; "%(ra,dec))
    d.set("dssstsci size 1.5 1.5 degrees;dssstsci %0.3f %0.3f; "%(ra,dec))
    d.set("regions file /Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FB/other_files/Frames/Guider%s.reg"%(field))
    
    a = Table.read("/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FB/other_files/targets/Master_target - GuidingStars.csv")
    a[a["Field"]==field].write("/tmp/test_%s.fits"%(field),overwrite=True)
    
    if field[:3]=="QSO":
        path = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FB/other_files/targets/targets_QSO.csv"
    elif field[:3]=="BQS":
        path = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FB/other_files/targets/QSO_mask_bright.csv"
    else:
        path = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FB/other_files/targets/targets_%s.csv"%(field)
        
    command = """catalog import TSV %s; catalog symbol color red ; catalog x RA ;
                                     catalog y DEC ; mode catalog"""%(path)
    try:
        d.set(command)
    except ValueError:
        pass
    
    command = """catalog import FITS /tmp/test_%s.fits ; catalog symbol color green ;  catalog x RA ; catalog y DEC ;"""%(field)#catalog symbol color green ; 
                                     
    try:
        d.set(command)
    except ValueError:
        pass
    # d.set("catalog close")

# d=DS9n()
# for field in ["F1","F2","F3","F4"][:1]:
# for field in ["QSO1","QSO2","QSO3","QSO4"][:]:
# for field in ["BQSO1","BQSO2","BQSO3"][:]:
for field in fields:
    DS9plot_field(field)