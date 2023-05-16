from astropy.table import Table
# d=DS9n()
names = d.get('plot').split()
name = names[0]
for i, name in enumerate(names):
    d.set("plot current %s " % (name))
    d.set("plot save /tmp/test_%s.dat" % (i))


d.set("plot all")
colors = ["red", "blue", "green", "black","orange"]*5
for i, (name, c) in enumerate(zip(names, colors)):
    filename = "/tmp/test_%s.dat" % (len(names)-i-1)
    # a = Table.read(filename,format='ascii')
    # a['col2'] = (a['col2']-a['col2'].min()) / (a['col2']-a['col2'].min()).ptp()
    # a.write(filename,overwrite=True,format='ascii',names=[None,None] )
    if len(Table.read(filename,format="ascii").colnames)==2:
        d.set("plot load %s xy" % (filename))
    elif len(Table.read(filename,format="ascii").colnames)==3:
        d.set("plot load %s xyey" % (filename))
    elif len(Table.read(filename,format="ascii").colnames)==4:
        d.set("plot load %s xyexey" % (filename))
    d.set("plot line color %s" % (c))

d.set("plot legend yes")
