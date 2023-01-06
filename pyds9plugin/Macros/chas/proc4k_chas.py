#! /usr/current/anaconda/bin/python
#
# Paul Martini (OSU) 
#
#  proc4k.py files 
#
# Perform the overscan subtraction and remove the relative gain 
# differences for a single R4K image or a list of R4K images. 
# Also works for MDM4K. 
#
# Steps: 
#  1. determine if input is a file or list of files
#  2. identify binning, size of overscan
#  3. remove overscan and trim
#  4. remove relative gain variations (TBD) 
#
#   8 Sep 2011: initial version for just bias subtraction 
#  12 Sep 2011: tested, adapted to run on MDM computers, work on MDM4K data
#  16 Sep 2011: added glob module
#   8 Feb 2012: fixed error in even/odd column definitions, added more 
# 		tracking and debugging information  
#
#-----------------------------------------------------------------------------

import string as str
import os 
from sys import argv, exit
import numpy as np 
import pyfits 
import glob
from tqdm import tqdm
# Version and Date

versNum = "1.1.0"
versDate = "2012-02-08"

############################
#### Define various routines
############################

scriptname = argv[0][str.rfind(argv[0], "/")+1::]

def usage(): 
  print "\nUsage for %s v%s (%s):" % (scriptname, versNum, versDate) 
  print "	%s file.fits [or file*.fits or file1.fits file2.fits ...]" % (scriptname) 
  print "\nWhere: file.fits, file*.fits, etc. are fits files\n" 

def parseinput():
  flags = []
  files = []
  # check for any command-line arguments and input files 
  for i in range(1,len(argv)): 
    if find(argv[i], "-") == 0: 
      flags.append(argv[i].strip("-"))
    else: 
      files.append(argv[i]) 
  # check that the input files exist
  for i in range(1,len(files)): 
    if os.path.isfile(files[i]) == 0:
      print "\n** ERROR: "+files[i]+" does not exist." 
      exit(1) 
  return files, flags

############################
#### Script starts here ####
############################

Debug=False
BiasSingle = 0
BiasRow = 1
BiasFit = 2
#BiasType = BiasRow
BiasType = BiasSingle
Gain = False		# keep as False until gain values are known 
R4K = True 

# Gain values for each amplifier [to be computed] 
r4k_gain_q1e = 1.0
r4k_gain_q1o = 1.0
r4k_gain_q2e = 1.0
r4k_gain_q2o = 1.0
r4k_gain_q3e = 1.0
r4k_gain_q3o = 1.0
r4k_gain_q4e = 1.0
r4k_gain_q4o = 1.0
mdm4k_gain_q1 = 1.0
mdm4k_gain_q2 = 1.0
mdm4k_gain_q3 = 1.0
mdm4k_gain_q4 = 1.0

# switch to more primitive (slower) code at MDM
AT_MDM = False
user = os.getlogin()
if str.find(user, 'obs24m') >= 0 or str.find(user, 'obs13m') >= 0: 
  AT_MDM = True 
AT_MDM = True 

files = []
for input in argv[1:]:  
  files = files + glob.glob(input) 

if len(files) == 0: 
  usage()
  exit(1) 
 
for file in files: 
  if os.path.isfile(file): 
    fitsfile = pyfits.open(file) 
    naxis1 = fitsfile[0].header['NAXIS1']
    naxis2 = fitsfile[0].header['NAXIS2']
    overscanx = fitsfile[0].header['OVERSCNX']
    overscany = fitsfile[0].header['OVERSCNY']	# should be 0 
    ccdxbin = fitsfile[0].header['CCDXBIN']
    ccdybin = fitsfile[0].header['CCDYBIN']	
    detector = fitsfile[0].header['DETECTOR']	
    telescope = fitsfile[0].header['TELESCOP'] 
    overscanx /= ccdxbin
    overscany /= ccdybin
    # OSMOS or direct? [useful for knowing if MIS keywords have values] 
    OSMOS = True
    if str.find(telescope, 'McGraw') >= 0: 
      OSMOS = False	# direct image with the 1.3m 
    #print file, naxis1, naxis2, overscanx, overscany, detector 
    print "Processing %s[%d:%d] OVERSCANX=%d OVERSCANY=%d from %s obtained at the %s" % (file, naxis1, naxis2, overscanx, overscany, detector, telescope)
    if overscanx*ccdxbin < 32: 
      print "Error: OVERSCNX=%d less than 32 in %s" % (overscanx, file) 
      exit(1)
    if overscany > 0: 
      print "Error: code not tested with OVERSCNY > 0!" 
      exit(1)
    if str.find(detector, 'R4K') < 0: 
      # if not R4K, assume MDM4K 
      R4K  = False
    #   IRAF units: 1:32, 33:556, 557:1080, 1081:1112
    # Python units: 0:31, 32:555, 556:1079, 1080:1111
    c1 = overscanx 		# 32   first image column counting from *zero*
    c2 = int(0.5*naxis1)-1	# 555  last image column on first half 
    c3 = c2+1			# 556  first image column on second half 
    c4 = naxis1-overscanx-1 	# 1079 last image column 
    r1 = overscany 		# 0    first image row 
    r2 = int(0.5*naxis2)-1	# 523  last image row on first half 
    r3 = r2+1			# 524  first image row on second half 
    r4 = naxis2-overscany-1  	# 1047 last image row 
    outnaxis1 = c4-c1+1		# 1048 columns in output, trimmed image 
    outnaxis2 = r4-r1+1		# 1048 rows in output, trimmed image
    collen = int(0.5*outnaxis1)	# number of rows in an image quadrant
    rowlen = int(0.5*outnaxis2)	# number of rows in an image quadrant

    #
    # Assumed layout: (ds9 perspective) 
    #
    #    q2	q4
    # 
    #	 q1	q3
    # 
    # each R4K quadrant has an even 'e' and an odd 'o' amplifier
    # 
    # print(0)
    if Debug: 
      print "Quadrants in IRAF pixels: " 
      print " q1: [%d:%d,%d:%d]" % (c1+1, c2+1, r1+1, r2+1) 
      print " q2: [%d:%d,%d:%d]" % (c1+1, c2+1, r3+1, r4+1) 
      print " q3: [%d:%d,%d:%d]" % (c3+1, c4+1, r1+1, r2+1) 
      print " q4: [%d:%d,%d:%d]" % (c3+1, c4+1, r3+1, r4+1) 
    ## Calculate the bias level for each amplifier
    data = fitsfile[0].data
    # identify the columns to use to calculate the bias level 
    # skip the first and last columns of the overscan 
    # changed to 'list' for hiltner due to primitive python version 
    starti = 4/ccdxbin
    if AT_MDM: 
      if R4K: 
        cols_over_q1e = list(np.arange(starti, overscanx-2, 2))
        cols_over_q1o = list(np.arange(starti+1, overscanx-2, 2)) 
        cols_over_q2e = cols_over_q1e
        cols_over_q2o = cols_over_q1o 
        cols_over_q3e = list(np.arange(naxis1-overscanx+starti, naxis1-2, 2))
        cols_over_q3o = list(np.arange(naxis1-overscanx+starti+1, naxis1-2, 2))
        cols_over_q4e = cols_over_q3e
        cols_over_q4o = cols_over_q3o 
        cols_q1e = list(np.arange(c1,c2,2))
        cols_q1o = list(np.arange(c1+1,c2+2,2))
        cols_q2e = cols_q1e
        cols_q2o = cols_q1o
        cols_q3e = list(np.arange(c3,c4,2))
        cols_q3o = list(np.arange(c3+1,c4+2,2))
        cols_q4e = cols_q3e
        cols_q4o = cols_q3o
      else: 
        cols_over_q1 = list(np.arange(starti, overscanx-2, 1))
        cols_over_q2 = cols_over_q1
        cols_over_q3 = list(np.arange(naxis1-overscanx+starti, naxis1-2, 1))
        cols_over_q4 = cols_over_q3
        cols_q1 = list(np.arange(c1,c2+1,1))
        cols_q2 = cols_q1
        cols_q3 = list(np.arange(c3,c4+1,1))
        cols_q4 = cols_q3
    else: 
      if R4K: 
        # identify the even and odd columns in the overscan
        cols_over_q1e = np.arange(starti, overscanx-starti, 2)
        cols_over_q1o = np.arange(starti+1, overscanx-starti, 2)
        cols_over_q2e = cols_over_q1e
        cols_over_q2o = cols_over_q1o 
        cols_over_q3e = np.arange(naxis1-overscanx+starti, naxis1-starti, 2)
        cols_over_q3o = np.arange(naxis1-overscanx+starti+1, naxis1-starti, 2)
        cols_over_q4e = cols_over_q3e
        cols_over_q4o = cols_over_q3o 
        # identify the even and odd columns in each quadrant 
        cols_q1e = np.arange(c1,c2,2)
        cols_q2e = cols_q1e
        cols_q1o = np.arange(c1+1,c2+2,2)
        cols_q2o = cols_q1o
        cols_q3e = np.arange(c3,c4,2)
        cols_q4e = cols_q3e
        cols_q3o = np.arange(c3+1,c4+2,2)
        cols_q4o = cols_q3o
      else: 
        cols_over_q1 = np.arange(starti, overscanx-2, 1)
        cols_over_q2 = cols_over_q1
        cols_over_q3 = np.arange(naxis1-overscanx+starti, naxis1-2, 1)
        cols_over_q4 = cols_over_q3
        cols_q1 = np.arange(c1,c2+1,1)
        cols_q2 = cols_q1
        cols_q3 = np.arange(c3,c4+1,1)
        cols_q4 = cols_q3
    if Debug: 
      print "Overscan columns: " 
      print "Q1/Q2 overscan even first and last columns:", cols_over_q1e[0], cols_over_q1e[-1], len(cols_over_q1e) 
      print "Q1/Q2 overscan odd first and last columns:", cols_over_q1o[0], cols_over_q1o[-1], len(cols_over_q1o) 
      print "Q3/Q4 overscan even first and last columns:", cols_over_q3e[0], cols_over_q3e[-1], len(cols_over_q3e) 
      print "Q3/Q4 overscan odd first and last columns:", cols_over_q3o[0], cols_over_q3o[-1], len(cols_over_q3o) 
    if Debug: 
      print "Image columns: "
      print "Q1/Q2 even first and last columns:", cols_q1e[0], cols_q1e[-1], len(cols_q1e), r1, r2, len(cols_q1e)
      print "Q1/Q2 odd first and last columns:", cols_q1o[0], cols_q1o[-1], len(cols_q1o), r1+rowlen, r2+rowlen, len(cols_q1o)
      print "Q3/Q4 even first and last columns:", cols_q3e[0], cols_q3e[-1], len(cols_q3e), r1, r2, len(cols_q3e)
      print "Q3/Q4 odd first and last columns:", cols_q3o[0], cols_q3o[-1], len(cols_q3o), r1+rowlen, r2+rowlen, len(cols_q3o)
    # create arrays with the median overscan vs. row for each amplifier
    if R4K: 
      bias_q1e = np.zeros(rowlen, dtype=float)
      bias_q1o = np.zeros(rowlen, dtype=float)
      bias_q2e = np.zeros(rowlen, dtype=float)
      bias_q2o = np.zeros(rowlen, dtype=float)
      bias_q3e = np.zeros(rowlen, dtype=float)
      bias_q3o = np.zeros(rowlen, dtype=float)
      bias_q4e = np.zeros(rowlen, dtype=float)
      bias_q4o = np.zeros(rowlen, dtype=float)
    else: 
      bias_q1 = np.zeros(rowlen, dtype=float)
      bias_q2 = np.zeros(rowlen, dtype=float)
      bias_q3 = np.zeros(rowlen, dtype=float)
      bias_q4 = np.zeros(rowlen, dtype=float)
    # calculate 1-D bias arrays for each amplifier
    for i in range(r1, r2+1, 1): 
      if R4K: 
        bias_q1e[i] = np.median(data[i,cols_over_q1e]) 	# data[rows, columns]
        bias_q1o[i] = np.median(data[i,cols_over_q1o]) 	
        bias_q2e[i] = np.median(data[i+rowlen,cols_over_q2e])
        bias_q2o[i] = np.median(data[i+rowlen,cols_over_q2o])
        bias_q3e[i] = np.median(data[i,cols_over_q3e])
        bias_q3o[i] = np.median(data[i,cols_over_q3o])
        bias_q4e[i] = np.median(data[i+rowlen,cols_over_q4e])
        bias_q4o[i] = np.median(data[i+rowlen,cols_over_q4o])
      else: #MDM4K 
        bias_q1[i] = np.median(data[i,cols_over_q1]) 	# data[rows, columns]
        bias_q2[i] = np.median(data[i+rowlen,cols_over_q2])
        bias_q3[i] = np.median(data[i,cols_over_q3])
        bias_q4[i] = np.median(data[i+rowlen,cols_over_q4])

    ########################################################################## 
    # Subtract the bias from the output
    ########################################################################## 

    if BiasType == BiasSingle: 
      OverscanKeyValue = 'BiasSingle' 
      suffix = 'b'
      # subtract a single bias value for each amplifier
      if R4K: 
        bq1e = np.median(bias_q1e) 
        bq1o = np.median(bias_q1o) 
        bq2e = np.median(bias_q2e) 
        bq2o = np.median(bias_q2o) 
        bq3e = np.median(bias_q3e) 
        bq3o = np.median(bias_q3o) 
        bq4e = np.median(bias_q4e) 
        bq4o = np.median(bias_q4o) 
        if AT_MDM: 
          for r in range(r1,r2+1): 
            for c in cols_q1e: 
              data[r,c] -= bq1e 
            for c in cols_q1o: 
              data[r,c] -= bq1o 
            for c in cols_q2e: 
              data[r+rowlen,c] -= bq2e 
            for c in cols_q2o: 
              data[r+rowlen,c] -= bq2o 
            for c in cols_q3e: 
              data[r,c] -= bq3e
            for c in cols_q3o: 
              data[r,c] -= bq3o
            for c in cols_q4e: 
              data[r+rowlen,c] -= bq4e
            for c in cols_q4o: 
              data[r+rowlen,c] -= bq4o
        else: 
          data[r1:r2+1,cols_q1e] -= bq1e
          data[r1:r2+1,cols_q1o] -= bq1o
          data[r3:r4+1,cols_q2e] -= bq2e
          data[r3:r4+1,cols_q2o] -= bq2o
          data[r1:r2+1,cols_q3e] -= bq3e
          data[r1:r2+1,cols_q3o] -= bq3o
          data[r3:r4+1,cols_q4e] -= bq4e
          data[r3:r4+1,cols_q4o] -= bq4o
      else: 
        bq1 = np.median(bias_q1) 
        bq2 = np.median(bias_q2) 
        bq3 = np.median(bias_q3) 
        bq4 = np.median(bias_q4) 

        rr = np.arange(r1,r2+1)
      	if AT_MDM: 
          # MODIFICATION VINCENT PICOUET
          # NOVEMBER 2022
          # IMPROVE SPEED FROM 30sec to 1sec
          data[rr.min():rr.max()+1,np.min(cols_q1):np.max(cols_q1)+1] -= int(bq1 )
          data[rr.min()+rowlen:rr.max()+1+rowlen,np.min(cols_q2):np.max(cols_q2)+1] -= int(bq2 )
          data[rr.min():rr.max()+1,np.min(cols_q3):np.max(cols_q3)+1] -= int(bq3 )
          data[rr.min()+rowlen:rr.max()+1+rowlen,np.min(cols_q4):np.max(cols_q4)+1] -= int(bq4 )

          # for i in tqdm(range(len(rr))):
          #   # print("r=",r)
          #   r=rr[i]
          #   for c in cols_q1: 
          #     data[r,c] -= bq1 
          #   for c in cols_q2: 
          #     data[r+rowlen,c] -= bq2 
          #   for c in cols_q3: 
          #     data[r,c] -= bq3
          #   for c in cols_q4: 
          #     data[r+rowlen,c] -= bq4
            # data[r,np.min(cols_q1):np.max(cols_q1)] -= int(bq1 )
            # data[r+rowlen,np.min(cols_q2):np.max(cols_q2)] -= int(bq2 )
            # data[r,np.min(cols_q3):np.max(cols_q3)] -= int(bq3 )
            # data[r+rowlen,np.min(cols_q4):np.max(cols_q4)] -= int(bq4 )

        else: 
          data[r1:r2+1,cols_q1] -= bq1
          data[r3:r4+1,cols_q2] -= bq2
          data[r1:r2+1,cols_q3] -= bq3
          data[r3:r4+1,cols_q4] -= bq4
    elif BiasType == BiasRow: 
      print("BiasType")
      # not implemented on Hiltner, for MDM4K, etc. 
      print "Warning: This mode has not been fully tested" 
      OverscanKeyValue = 'BiasRow' 
      # subtract a bias value for each row of each amplifier 
      #print r1, r2, len(bias_q1e) 
      suffix = 'br' 
      for i in range(r1, r2, 1): 
        print(i)
        data[i,cols_q1e] -= bias_q1e[i]
        data[i,cols_q1o] -= bias_q1o[i]
        data[i+rowlen,cols_q2e] -= bias_q2e[i]
        data[i+rowlen,cols_q2o] -= bias_q2o[i]
        data[i,cols_q3e] -= bias_q3e[i]
        data[i,cols_q3o] -= bias_q3o[i]
        data[i+rowlen,cols_q4e] -= bias_q4e[i]
        data[i+rowlen,cols_q4o] -= bias_q4o[i]
    elif BiasType == BiasFit: 
      OverscanKeyValue = 'BiasFit' 
      print "Error: Have not implemented a fit to the bias yet. Please use BiasSingle" 
    else: 
      print "Error: Bias subtraction type not parsed correctly" 
      # exit(1) 

    ########################################################################## 
    # Apply the gain correction  [not yet implemented] 
    ########################################################################## 

    if Gain: 
      if R4K: 
        if AT_MDM: 
          for r in range(r1,r2+1): 
            for c in cols_q1e: 
              data[r,c] -= r4k_gain_q1e 
            for c in cols_q1o: 
              data[r,c] -= r4k_gain_q1o 
            for c in cols_q2e: 
              data[r+rowlen,c] -= r4k_gain_q2e 
            for c in cols_q2o: 
              data[r+rowlen,c] -= r4k_gain_q2o 
            for c in cols_q2o: 
              data[r,c] -= r4k_gain_q3e
            for c in cols_q2o: 
              data[r,c] -= r4k_gain_q3o
            for c in cols_q2o: 
              data[r+rowlen,c] -= r4k_gain_q4e
            for c in cols_q2o: 
              data[r+rowlen,c] -= r4k_gain_q4o
        else: 
          data[r1:r2,cols_q1e] /= r4k_gain_q1e 
          data[r1:r2,cols_q1o] /= r4k_gain_q1o 
          data[r3:r4,cols_q2e] /= r4k_gain_q2e 
          data[r3:r4,cols_q2o] /= r4k_gain_q2o 
          data[r1:r2,cols_q3e] /= r4k_gain_q3e 
          data[r1:r2,cols_q3o] /= r4k_gain_q3o 
          data[r3:r4,cols_q4e] /= r4k_gain_q4e 
          data[r3:r4,cols_q4o] /= r4k_gain_q4o 
      else: 
        if AT_MDM: 
          for r in range(r1,r2+1): 
            for c in cols_q1: 
              data[r,c] /= mdm4k_gain_q1
            for c in cols_q2: 
              data[r+rowlen,c] /= mdm4k_gain_q2
            for c in cols_q2: 
              data[r,c] /= mdm4k_gain_q3
            for c in cols_q2: 
              data[r+rowlen,c] /= mdm4k_gain_q4
        else: 
          data[r1:r2,cols_q1] /= mdm4k_gain_q1 
          data[r3:r4,cols_q2] /= mdm4k_gain_q2 
          data[r1:r2,cols_q3] /= mdm4k_gain_q3 
          data[r3:r4,cols_q4] /= mdm4k_gain_q4 


    ########################################################################## 
    # Write the output file 
    ########################################################################## 

    fitsfile[0].data = data[r1:r4+1,c1:c4+1] 
    OverscanKeyComment = 'Overscan by proc4k.py v%s (%s)' % (versNum, versDate) 
    GainKeyValue = 'Relative'
    GainKeyComment = 'Gain removed by proc4k.py' 
    #BiasKeyValue = '%s' % (versNum) 
    #BiasKeyComment = 'Gain removed by proc4k.py' 

    if OSMOS: 	# prevent a pyfits error if these are not assigned values 
      try: 
        fitsfile[0].header['MISFILT'] = -1
        fitsfile[0].header['MISFLTID'] = -1
      except: 
        if Debug: 
          print "Note: MISFILT and MISFLTID keywords not found" 

    # fitsfile[0].header.update('BIASPROC', OverscanKeyValue, OverscanKeyComment) 
    #fitsfile[0].header.update('BIASVER', BiasKeyValue, BiasKeyComment) 
    if R4K: 
      fitsfile[0].header['BIASQ1E']=( bq1e, 'Bias subtracted from Q1E') 
      fitsfile[0].header['BIASQ1O']=( bq1o, 'Bias subtracted from Q1O') 
      fitsfile[0].header['BIASQ2E']=( bq2e, 'Bias subtracted from Q2E') 
      fitsfile[0].header['BIASQ2O']=( bq2o, 'Bias subtracted from Q2O') 
      fitsfile[0].header['BIASQ3E']=( bq3e, 'Bias subtracted from Q3E') 
      fitsfile[0].header['BIASQ3O']=( bq3o, 'Bias subtracted from Q3O') 
      fitsfile[0].header['BIASQ4E']=( bq4e, 'Bias subtracted from Q4E') 
      fitsfile[0].header['BIASQ4O']=( bq4o, 'Bias subtracted from Q4O') 
    else: 
      # fitsfile[0].header.update('BIASQ1', bq1, 'Bias subtracted from Q1') 
      # fitsfile[0].header.update('BIASQ2', bq2, 'Bias subtracted from Q2') 
      # fitsfile[0].header.update('BIASQ3', bq3, 'Bias subtracted from Q3') 
      # fitsfile[0].header.update('BIASQ4', bq4, 'Bias subtracted from Q4') 
      pass
    if Gain: 
      if R4K: 
        fitsfile[0].header['GAINPROC'] = (  GainKeyValue, GainKeyComment) 
        fitsfile[0].header['GainQ1'  ] = (r4k_gain_q1, 'Gain for Q1') 
        fitsfile[0].header['GainQ2'  ] = (r4k_gain_q2, 'Gain for Q2') 
        fitsfile[0].header['GainQ3'  ] = (r4k_gain_q3, 'Gain for Q3') 
        fitsfile[0].header['GainQ4'  ] = (r4k_gain_q4, 'Gain for Q4') 

    outfile = file[:str.find(file, '.fits')]+suffix+'.fits' 
    if os.path.isfile(outfile): 
      print "  Warning: Overwriting pre-existing file %s" % (outfile) 
      os.remove(outfile) 
    fitsfile.writeto(outfile)
    fitsfile.close() 

# print "%s Done" % (argv[0]) 
print "%s Done" % (scriptname) 

