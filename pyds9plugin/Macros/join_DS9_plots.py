# d=DS9n()
names = d.get('plot').split()
name = names[0]
for i, name in enumerate(names):
    d.set("plot current %s " % (name))
    d.set("plot save /tmp/test_%s.dat" % (i))


d.set("plot all")
colors = ["red", "blue", "green", "black"]
for i, (name, c) in enumerate(zip(names, colors)):
    filename = "/tmp/test_%s.dat" % (len(names)-i-1)
    # a = Table.read(filename,format='ascii')
    # a['col2'] = (a['col2']-a['col2'].min()) / (a['col2']-a['col2'].min()).ptp()
    # a.write(filename,overwrite=True,format='ascii',names=[None,None] )
    d.set("plot load  /tmp/test_%s.dat xy" % (len(names)-i-1))
    d.set("plot line color %s" % (c))

d.set("plot legend yes")
