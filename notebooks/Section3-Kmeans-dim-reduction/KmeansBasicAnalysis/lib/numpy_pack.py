import numpy as np

"""Code for packing and unpacking a numpy array into a byte array.
   the array is flattened if it is not 1D.
   This is intended to be used as the interface for storing 
   
   This code is intended to be used to store numpy array as fields in a dataframe and then store the 
   dataframes in a parquet file.
"""

def packArray(a):
    """
    pack a numpy array into a bytearray that can be stored as a single 
    field in a spark DataFrame

    :param a: a numpy ndarray 
    :returns: a bytearray
    :rtype:

    """
    if type(a)!=np.ndarray:
        raise Exception("input to packArray should be numpy.ndarray. It is instead "+str(type(a)))
    return bytearray(a.tobytes())

def unpackAndScale(row):
    """ Unpack bytearray, then if measurement is in ['TMIN','TMAX','TOBS']
    then divide by 10 to get celsius
    """
    if '_S' in row.Measurement:
        v=unpackArray(row.Values,data_type=np.float16)
    else:
        v=unpackArray(row.Values)
    if(row.Measurement in ['TMIN','TMAX','TOBS']):
        v=v/10
    return v

def unpackArray(x,data_type=np.int16):
    """
    unpack a bytearray into a numpy.ndarray, values Smaller than -990 (nominally -999) are mapped to np.nan

    :param x: a bytearray
    :param data_type: The dtype of the array. This is important because if determines how many bytes go into each entry in the array.
    :returns: a numpy array of float16
    :rtype: a numpy ndarray of dtype data_type.

    """
    V=np.frombuffer(x,dtype=data_type)
    V=np.array(V,dtype=np.float16)
    V[V<-990]=np.nan
    return V
