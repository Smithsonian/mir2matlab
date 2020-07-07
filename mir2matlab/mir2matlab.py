import numpy as np
import scipy.io
import os, sys, struct, mmap, argparse, shutil, time, glob
from astropy.time import Time
from datetime import datetime

########################################################
# MIR structure definitions

inDataType = np.dtype([
    ('traid', np.int32),
    ('inhid', np.int32),
    ('ints', np.int32),
    ('az', np.single),
    ('el', np.single),
    ('ha', np.single),
    ('iut', np.int16),
    ('iref_time', np.int16),
    ('dhrs', np.double),
    ('vc', np.single),
    ('sx', np.double),
    ('sy', np.double),
    ('sz', np.double),
    ('rinteg', np.single),
    ('proid', np.int32),
    ('souid', np.int32),
    ('isource', np.int16),
    ('ivrad', np.int16),
    ('offx', np.single),
    ('offy', np.single),
    ('ira', np.int16),
    ('idec', np.int16),
    ('rar', np.double),
    ('decr', np.double),
    ('epoch', np.single),
    ('size', np.single),
    ('vrra', np.single),
    ('vrdec', np.single),
    ('lst', np.single),
    ('iproject', np.int16),
    ('tile', np.int16),
    ('obsmode', np.uint8),
    ('obsflag', np.uint8),
    ('spareshort', np.int16),
    ('spareint6', np.int32),
    ('yIGFreq1', np.double),
    ('yIGFreq2', np.double),
    ('sflux', np.double),
    ('ara', np.double),
    ('adec', np.double),
    ('mjd', np.double),
    ]);

engDataType = np.dtype([
    ('antennaNumber', np.int32),
    ('padNumber', np.int32),
    ('antennaStatus', np.int32),
    ('trackStatus', np.int32),
    ('commStatus', np.int32),
    ('inhid', np.int32),
    ('ints', np.int32),
    ('dhrs', np.double),
    ('ha', np.double),
    ('lst', np.double),
    ('pmdaz', np.double),
    ('pmdel', np.double),
    ('tiltx', np.double),
    ('tilty', np.double),
    ('actual_az', np.double),
    ('actual_el', np.double),
    ('azoff', np.double),
    ('eloff', np.double),
    ('az_tracking_error', np.double),
    ('el_tracking_error', np.double),
    ('refraction', np.double),
    ('chopper_x', np.double),
    ('chopper_y', np.double),
    ('chopper_z', np.double),
    ('chopper_angle', np.double),
    ('tsys', np.double),
    ('tsys_rx2', np.double),
    ('ambient_load_temperature', np.double)
    ]);

blDataType = np.dtype([
    ('blhid', np.int32),
    ('inhid', np.int32),
    ('isb', np.int16),
    ('ipol', np.int16),
    ('ant1rx', np.int16),
    ('ant2rx', np.int16),
    ('pointing', np.int16),
    ('irec', np.int16),
    ('u', np.single),
    ('v', np.single),
    ('w', np.single),
    ('prbl', np.single),
    ('coh', np.single),
    ('avedhrs', np.double),
    ('ampave', np.single),
    ('phaave', np.single),
    ('blsid', np.int32),
    ('iant1', np.int16),
    ('iant2', np.int16),
    ('ant1TsysOff', np.int32),
    ('ant2TsysOff', np.int32),
    ('iblcd', np.int16),
    ('ble', np.single),
    ('bln', np.single),
    ('blu', np.single),
    ('spareint1', np.int32),
    ('spareint2', np.int32),
    ('spareint3', np.int32),
    ('spareint4', np.int32),
    ('spareint5', np.int32),
    ('spareint6', np.int32),
    ('sparedbl1', np.double),
    ('sparedbl2', np.double),
    ('sparedbl3', np.double),
    ('sparedbl4', np.double),
    ('sparedbl5', np.double),
    ('sparedbl6', np.double)
    ]);

spDataType = np.dtype([
    ('sphid', np.int32),
    ('blhid', np.int32),
    ('inhid', np.int32),
    ('igq', np.int16),
    ('ipq', np.int16),
    ('iband', np.int16),
    ('ipstate', np.int16),
    ('tau0', np.single),
    ('vel', np.double),
    ('vres', np.single),
    ('fsky', np.double),
    ('fres', np.single),
    ('gunnLO', np.double),
    ('cabinLO', np.double),
    ('corrLO1', np.double),
    ('corrLO2', np.double),
    ('integ', np.single),
    ('wt', np.single),
    ('flags', np.int32),
    ('vradcat', np.single),
    ('nch', np.int16),
    ('nrec', np.int16),
    ('dataoff', np.int32),
    ('rfreq', np.double),
    ('corrblock', np.int16),
    ('corrchunk', np.int16),
    ('correlator', np.int32),
    ('spareint2', np.int32),
    ('spareint3', np.int32),
    ('spareint4', np.int32),
    ('spareint5', np.int32),
    ('spareint6', np.int32),
    ('sparedbl1', np.double),
    ('sparedbl2', np.double),
    ('sparedbl3', np.double),
    ('sparedbl4', np.double),
    ('sparedbl5', np.double),
    ('sparedbl6', np.double)
    ]);

codesDataType = np.dtype([
    ('v_name', np.character, 12),
    ('icode', np.int16),
    ('code', np.character, 26),
    ('ncode', np.int16)
    ]);

weDataType = np.dtype([
    ('scanNumber', np.int32),
    ('flags', np.int32, 11),
    ('N', np.single, 11),
    ('Tamb', np.single, 11),
    ('pressure', np.single, 11),
    ('humid', np.single, 11),
    ('windSpeed', np.single, 11),
    ('windDir', np.single, 11),
    ('h2o', np.single, 11)
    ]);

# end structure defs
########################################################


def readInData(dataPath):
  inData = np.fromfile(dataPath + '/in_read',dtype=inDataType);
  return inData;

def readEngData(dataPath):
  engData = np.fromfile(dataPath + '/eng_read',dtype=engDataType);
  return engData;

def readBlData(dataPath):
  blData = np.fromfile(dataPath + '/bl_read',dtype=blDataType);
  return blData;

def readSpData(dataPath):
  spData = np.fromfile(dataPath + '/sp_read',dtype=spDataType);
  return spData;

def readCodesData(dataPath):
  codesData = np.fromfile(dataPath + '/codes_read',dtype=codesDataType);
  return codesData;

def readWeData(dataPath):
  weData = np.fromfile(dataPath + '/we_read',dtype=weDataType);
  return weData;

def scanIntStart(dataPath):
  fSize = os.path.getsize(dataPath + '/sch_read');
  visFile = open(dataPath + '/sch_read', 'rb')

  dataOffset = 0;
  lastOffset = 0;
  inOffsetDict = {}
  while dataOffset < fSize:
    intVals = np.fromfile(visFile, dtype=np.dtype([('inhid', np.int32),('insize', np.int32)]), count=1, offset=lastOffset);
    inOffsetDict[intVals['inhid'][0]] = (intVals['insize'][0], dataOffset + 8);
    lastOffset = intVals['insize'][0];
    dataOffset += lastOffset + 8;

  visFile.close()
  return inOffsetDict;

def loadIntVis(dataPath,inOffsetDict):
  dataDict = {};
  visFile = open(dataPath + '/sch_read', 'rb');
  lastOffset = 0;
  for indKey in sorted (inOffsetDict.keys()):
    nVals = inOffsetDict[indKey][0]//2;
    lastOffset = inOffsetDict[indKey][1] - lastOffset;
    print(indKey,nVals,lastOffset,inOffsetDict[indKey][1])
    dataDict[indKey] = np.fromfile(visFile, dtype=np.int16, count=nVals, offset=lastOffset);
    lastOffset = inOffsetDict[indKey][1] + nVals + nVals;

  visFile.close()
  return dataDict;

def checkDataInteg(dataPath,checkVis=False):
  checkResult = True
  try:
    _ = readInData(dataPath)
    _ = readEngData(dataPath)
    _ = readBlData(dataPath)
    _ = readSpData(dataPath)
    _ = readCodesData(dataPath)
    _ = readWeData(dataPath)
    if checkVis:
      # do something more here
      1+1
  except:
    checkResult = False
  return checkResult

