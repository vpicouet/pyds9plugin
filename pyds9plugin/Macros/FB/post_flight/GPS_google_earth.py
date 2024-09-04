
from astropy.table import Table 
# data = Table.read("/Users/Vincent/Github/FB-dashboard/FlightPrincipalCatalog_v5_tronc.csv")
# data = Table.read("/Users/Vincent/Library/CloudStorage/GoogleDrive-vp2376@columbia.edu/.shortcut-targets-by-id/1ZgB7kY-wf7meXrq8v-1vIzor75aRdLDn/FIREBall-2/FB2_2023/Flight/CNES/Fichiers_Pour_La_Science_FLIGHT_2023_tronc10.csv")
data = Table.read("/Users/Vincent/Library/CloudStorage/OneDrive-CaliforniaInstituteofTechnology/Safari/Fichiers_Pour_La_Science_FLIGHT_2023.csv")
f = open('/tmp/flight.kml', 'w')

#Writing the kml file.
f.write("<?xml version='1.0' encoding='UTF-8'?>\n")
f.write("<kml xmlns='http://earth.google.com/kml/2.2'>\n")
f.write("<Document>\n")
f.write("<Placemark>\n")
f.write("   <name>flight</name>\n")
f.write("   <LineString>\n")
f.write("       <extrude>1</extrude>\n")
f.write("       <altitudeMode>absolute</altitudeMode>\n")
f.write("       <coordinates>\n")
for i in range(0,len(data['Latitude']),1):  #Here I skip some data
    f.write("        "+str(data['Longi'][i]) + ","+ str(data['Latitude'][i]) + "," + str(data['Altitude'][i]) +"\n")    
f.write("       </coordinates>\n")
f.write("   </LineString>\n")
f.write("</Placemark>\n")
f.write("</Document>")
f.write("</kml>\n")
f.close()


