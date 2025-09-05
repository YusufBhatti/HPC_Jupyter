# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:19:59 2020

@author: ybh10
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 03:19:06 2018

Contains:
    save_dict
    load_dict
    times
    t_lon_array
    areaweight
    make_climatology
    make_climatology2
    make_clim_std
    get_aeronet_aod_550nm_monthly_mean
    get_aeronet_aod_550nm
    toYearFraction
    interpolate
    interpolate_new
    get_seasonal_var
    find_index
    calc_trop_o3_column
    shiftedColorMap
    polar_stereographic_plot_contourf
    polar_stereographic_plot_contourf_and_contour
    calc_density
    check_lat_lon
    make_seasonal_mean

@author: Yusuf Bhatti
"""
from six.moves import cPickle as pickle
import numpy as np
from netCDF4 import Dataset
import netCDF4 as nc
from datetime import datetime as dt
import time
#import scipy.io
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import scipy.interpolate as interpolate
import xarray as xr

def average_group_std(data):
    # Square the standard deviations
    squared_std = data**2
    # Compute the mean of squared standard deviations across the groups
    mean_squared_std = squared_std.mean(dim='sample')
    # Take the square root to get the Average S.D.
    average_std_dev = np.sqrt(mean_squared_std)
    return average_std_dev
    
def calc_pdf(data, left, right, nbins):
    '''
    This is a function to calculate pdf
data: data to calculate PDF must be 1D
left: left bound of data
right: right bound of data
    nbins: number of bins
    '''
    data = data[~np.isnan(data)]
    right = right-0.1
    
    bins = np.linspace(left, right, nbins)
    
    frequency = np.zeros(len(bins))
    
    for i in range(0,nbins):
        if np.isnan(np.nanmean(data)) == False:
            if i < nbins-1:
                mask = (bins[i]<= data) & (data < bins[i+1])
                frequency[i] = len(data[mask])/len(data)
            else:
                mask = data >= bins[i]
                frequency[i] = len(data[mask])/len(data)
        else:
            frequency[i] = np.nan
    return(bins,frequency)

def calc_cdf(perd):
    cpd = np.zeros(len(perd))
    for i in range(0,len(perd)):
        cpd[i] = perd[i]+ cpd[i-1]
    return(cpd)


def normalize(x):
    return (x - x.min())/(x.max() - x.min())


def save_dict(di_, filename_): # TO SAVE A DICTIONARY INTO .NPY
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_): # TO LOAD A DICTIONARY INTO .NPY
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def times(days,months,year): # Creates a list of conjoined dates
    time=[]
    if np.size(year)>1:
        if np.size(months)>1:
            for yr in (year):
                for mon in (months):
                    for day in (days):
                        print('{:02d}{}{}'.format(day,mon,yr))
                        time.append(('{:02d}{}{}'.format(day,mon,yr)))
        if np.size(months)==1:
            for yr in (year):
                for day in (days):
                    print('{:02d}{}{}'.format(day,months,yr))
                    time.append(('{:02d}{}{}'.format(day,months,yr)))
    if np.size(year)==1:
        if np.size(months)>1:
            for mon in (months):
                for day in (days):
                    print('{:02d}{}{}'.format(day,mon,year))
                    time.append(('{:02d}{}{}'.format(day,mon,year)))
        if np.size(months)==1:
            for day in (days):
                print('{:02d}{}{}'.format(day,months,year))
                time.append(('{:02d}{}{}'.format(day,months,yr)))
    return time


def monthly_iris(x): # Creates NC files from iris. #IN PROGRESS TO CHANGE!!
    constraint=iris.AttributeConstraint(STASH=Daily[x])
    cube = iris.load(filename,constraint)
    iris.save(cube, '/home/ybh10/DMS_Emissions/{}.nc'.format(x))
    print(('{}a.p{}2003{}.pp & {}').format(suite,use,months[i],Daily[x]))    
    print(cube)
    print('/home/ybh10/DMS_Emissions/{}.nc'.format(x))
    return monthly_iris

def month_shift(data): # Shifts months from Jan - Dec, to July - June
    First=[]
    Last=[]
    for i in np.arange(0,6,1):  ### RE-ORGANISE the months, to make it start from July --> Dec, then Dec to Jun...
        if data.ndim==2:
            Last.append(data[i,:])
            First.append(data[i,:])
            Months=np.concatenate((First,Last))
            Diff_Data_Months=Months.transpose(1,0) ## Re-Shape the array to make the RE-ORGANISED months into zonal 

        if data.ndim==3:
            Last.append(data[i,:,:])
            First.append(data[i+6,:,:])
            Months=np.concatenate((First,Last))
            Diff_Data_Months=Months.transpose(1,2,0) ## Re-Shape the array to make the RE-ORGANISED months into zonal 
    output = Diff_Data_Months

    return output


###############################################################################
def t_lon(input_lon):
    #Reorders longitude from 0:360 to -180:180
    h = np.int(np.size(input_lon)/2)
    t_lon = np.mod(input_lon+180,360)-180
    output_lon = np.concatenate((t_lon[h:],t_lon[:h]),axis=0)
    
    return output_lon

def t_lon_360(input_lon):
    #Reorders longitude from -180:180 to 0:360 
    h = np.int(np.size(input_lon)/2)
    t_lon = np.mod(input_lon+360,360)
    output_lon = np.concatenate((t_lon[h:],t_lon[:h]),axis=0)
    
    return output_lon
###############################################################################    
def t_lon_array(input_lon,input_array):
    #Reorders an array so longitudes from 0:360 go to -180:180 (assumes that lon is the last dimension)
    h = np.int(np.size(input_lon)/2)
    
    if input_array.ndim==2:
        output_array = np.concatenate((input_array[:,h:],input_array[:,:h]),axis=1)
    if input_array.ndim==3:
        output_array = np.concatenate((input_array[:,:,h:],input_array[:,:,:h]),axis=2)
    if input_array.ndim==4:
        output_array = np.concatenate((input_array[:,:,:,h:],input_array[:,:,:,:h]),axis=3)
    
    return output_array

def t_lon_array_360(input_lon,input_array):
    #Reorders an array so longitudes from -180:180 go to 0:360 (assumes that lon is the last dimension)
    h = np.int(np.size(input_lon))
    
    if input_array.ndim==2:
        output_array = np.concatenate((input_array[:,h:],input_array[:,:h]),axis=1)
    if input_array.ndim==3:
        output_array = np.concatenate((input_array[:,:,h:],input_array[:,:,:h]),axis=2)
    if input_array.ndim==4:
        output_array = np.concatenate((input_array[:,:,:,h:],input_array[:,:,:,:h]),axis=3)
    
    return output_array
###############################################################################
def areaweight_numpy(x,latitude):
    #Area weights variable x by the cosine of the latitude.
    cos_lat = np.cos(latitude*np.pi/180)
    
    #assume two dimensions for now, LR, 1/3/17
	#add for three dimensions, LR, 8/1/18
    s = x.shape
    n = x.ndim
	
    if n==2 and s[0]==np.size(latitude): 
        cost3m = np.empty((s[0],s[1])); cost3m[:,:] = np.nan
        for i in range(0,s[1]):
            cost3m[:,i]= cos_lat
        m = np.nansum(np.nansum(x*cost3m,1),0)/np.nansum(np.nansum(cost3m,1),0) #a lat-lon array
		
    elif n==2 and s[0]!=np.size(latitude):
        cost3m = np.empty((s[0],s[1])); cost3m[:,:] = np.nan
        for i in range(0,s[0]):
            cost3m[i,:]= np.squeeze(cos_lat)
        m = np.nansum(x*cost3m,1)/np.nansum(cost3m,1) #a lev-lat or time-lat array
        
    elif n==3:
        cost3m = np.empty((s[0],s[1],s[2])); cost3m[:,:,:] = np.nan
        for i in range(0,s[0]):
            for j in range(0,s[2]):
                cost3m[i,:,j] = np.squeeze(cos_lat)
        m = np.nansum(np.nansum(x*cost3m,2),1)/np.nansum(np.nansum(cost3m,2),1)	#lat is the middle dimension, e.g. time-lat-lon or lev-lat-lon.	
        
    elif n==4:
        cost3m = np.empty((s[0],s[1],s[2],s[3])); cost3m[:,:,:,:] = np.nan
        for i in range(0,s[0]):
            for j in range(0,s[1]):
                for k in range(0,s[3]):
                    cost3m[i,j,:,k] = np.squeeze(cos_lat)
        m = np.nansum(np.nansum(x*cost3m,3),2)/np.nansum(np.nansum(cost3m,3),2)	#time-lev-lat-lon	
	
    elif n==5:
        cost3m = np.empty((s[0],s[1],s[2],s[3],s[4])); cost3m[:,:,:,:,:] = np.nan
        for i in range(0,s[0]):
            for j in range(0,s[1]):
                for k in range(0,s[2]):
                    for d in range (0,s[4]):
                        cost3m[i,j,k,:,d] = np.squeeze(cos_lat)
        m = np.nansum(np.nansum(x*cost3m,4),3)/np.nansum(np.nansum(cost3m,4),3)	#time-lev-lat-lon	
       
    elif n==6:
        cost3m = np.empty((s[0],s[1],s[2],s[3],s[4],s[5])); cost3m[:,:,:,:,:] = np.nan
        for i in range(0,s[0]):
            for j in range(0,s[1]):
                for k in range(0,s[2]):
                    for ll in range(0,s[3]):
                        for d in range (0,s[5]):
                            cost3m[i,j,k,ll,:,d] = np.squeeze(cos_lat)
        m = np.nansum(np.nansum(x*cost3m,5),4)/np.nansum(np.nansum(cost3m,5),4)	#time-lev-lat-lon	

    output = m
    return output


# def areaweight(x, latitude):
#     cos_lat = np.cos(latitude * np.pi / 180)
#     lat_dim = list(x.dims)[-2]
#     lon_dim = list(x.dims)[-1]

#     # Reshape cos_lat to match the shape of x
#     s = x.shape
#     if len(s) == 2:
#         cos_lat_reshaped = xr.DataArray(cos_lat, dims=[lat_dim]).expand_dims({lon_dim: s[-1]}, axis=-1)
#     elif len(s) == 3:
#         first_dim = list(x.dims)[0]
#         cos_lat_reshaped = xr.DataArray(cos_lat, dims=[lat_dim]).expand_dims({first_dim: s[0], lon_dim: s[-1]}, axis=(0, 2))
#     elif len(s) == 4:
#         first_dim = list(x.dims)[0]
#         second_dim = list(x.dims)[1]
#         cos_lat_reshaped = xr.DataArray(cos_lat, dims=[lat_dim]).expand_dims({first_dim: s[0], second_dim: s[1], lon_dim: s[-1]}, axis=(0, 1, 3))
#     elif len(s) == 5:
#         first_dim = list(x.dims)[0]
#         second_dim = list(x.dims)[1]
#         third_dim = list(x.dims)[2]
#         cos_lat_reshaped = xr.DataArray(cos_lat, dims=[lat_dim]).expand_dims({first_dim: s[0], second_dim: s[1], third_dim: s[2], lon_dim: s[-1]}, axis=(0, 1, 2, 4))
#     elif len(s) == 6:
#         first_dim = list(x.dims)[0]
#         second_dim = list(x.dims)[1]
#         third_dim = list(x.dims)[2]
#         fourth_dim = list(x.dims)[3]
#         cos_lat_reshaped = xr.DataArray(cos_lat, dims=[lat_dim]).expand_dims({first_dim: s[0], second_dim: s[1], third_dim: s[2], fourth_dim: s[3], lon_dim: s[-1]}, axis=(0, 1, 2, 3, 5))

#     # Apply weights without modifying the original latitude coordinate
#     m = (x * cos_lat_reshaped).sum(lon_dim).sum(lat_dim) / cos_lat_reshaped.sum(lon_dim).sum(lat_dim)
#     return m


def areaweight(x, latitude):
    """
    Compute the area-weighted mean of an xarray DataArray.

    Parameters:
    x (xarray.DataArray): Input data with latitude and longitude dimensions.
    latitude (xarray.DataArray or numpy.ndarray): Latitude values in degrees.

    Returns:
    xarray.DataArray: Area-weighted mean.
    """
    # Compute cosine of latitude (weights)
    cos_lat = np.cos(np.deg2rad(latitude))  # Convert degrees to radians and compute cosine

    # Identify latitude and longitude dimensions
    lat_dim = [dim for dim in x.dims if dim in ['lat', 'latitude']][0]  # Flexible dimension naming
    lon_dim = [dim for dim in x.dims if dim in ['lon', 'longitude']][0]

    # Reshape cos_lat to match the shape of x
    # Use xr.DataArray broadcasting to avoid explicit shape handling
    cos_lat_reshaped = xr.DataArray(cos_lat, dims=[lat_dim]).broadcast_like(x)

    # Mask weights where data is NaN
    masked_weights = cos_lat_reshaped.where(~np.isnan(x))

    # Normalize weights to sum only over valid data
    normalized_weights = masked_weights / masked_weights.sum(dim=[lat_dim, lon_dim])

    # Apply area-weighted mean
    weighted_mean = (x * normalized_weights).sum(dim=[lat_dim, lon_dim])

    return weighted_mean

def areaweighted_std(x, latitude):
    # Calculate the area-weighted mean of x
    
    area_weight = areaweight_numpy(x, latitude).data
    if area_weight.ndim > 0:
        mean = np.nanmean(area_weight)
        mean = np.array(mean)
    else:
        mean=area_weight
    # Subtract the mean from x to get the deviation of each element from the mean
    deviation = x - mean

    # Square the deviation to get the squared deviation of each element from the mean
    squared_deviation = deviation**2

    # Calculate the area-weighted mean of the squared deviations
    mean_squared_deviation = areaweight_numpy(squared_deviation, latitude)

    # Take the square root of the mean squared deviation to get the area-weighted standard deviation of x
    std = np.sqrt(mean_squared_deviation)
    if std.ndim>0:
        std=np.nanmean(std)
    return std,mean,area_weight
###############################################################################

def percentage_difference(x1,x2):
    return ((x2/x1)-1)*100
###############################################################################

def make_clim_std(x):
    #Organises x (single dimension array) into years and months, then
    #calculates the standard deviation over months to make a climatology.
    
    n = int(len(x)/12) #find number of years
    rearr = np.zeros((n,12),np.float64)    
    for i in range(0,n):
        rearr[i,:] = (x[i*12:(12*(i+1))])

    x = np.nanstd(rearr)
    return x
###############################################################################
def make_climatology(x):
    #Organises x (single dimension array) into years and months, then
    #averages over months to make a climatology.
    
    n = int(len(x)/12) #find number of years
    #rearr = np.zeros((n,12),np.float64)    
    rearr = np.empty((n,12)); rearr[:,:] = np.nan
    for i in range(0,n):
        rearr[i,:] = (x[i*12:(12*(i+1))])

    x = np.nanmean(rearr,0)
    return x

###############################################################################
def make_climatology2(x):
    #Loops over lat and lon, and organises x. the first dimension, into years and months, then
    #averages over months to make a climatology.
    shape = np.shape(x)
    ntime = shape[0]; nlat= shape[1]; nlon = shape[2]
    
    n = np.int(ntime/12) #find number of years
    
    data = np.empty((12,nlat,nlon)); data[:,:,:] = np.nan
    for la in range(0,nlat):
        for lo in range(0,nlon):
            rearr = np.empty((n,12)); rearr[:,:] = np.nan
            for i in range(0,n):
                rearr[i,:] = (x[i*12:(12*(i+1)),la,lo])

            data[:,la,lo] = np.nanmean(rearr,0)
    return data

###############################################################################
def make_clim_std2(x):
    #Loops over lat and lon, and organises x. the first dimension, into years and months, then
    #averages over months to calculate the standard deviation.
    
    shape = np.shape(x)
    ntime = shape[0]; nlat= shape[1]; nlon = shape[2]
    
    n = np.int(ntime/12) #find number of years
    
    data = np.empty((12,nlat,nlon)); data[:,:,:] = np.nan
    for la in range(0,nlat):
        for lo in range(0,nlon):
            rearr = np.empty((n,12)); rearr[:,:] = np.nan
            for i in range(0,n):
                rearr[i,:] = (x[i*12:(12*(i+1)),la,lo])

            data[:,la,lo] = np.nanstd(rearr,axis=0,ddof=1)
    return data

###############################################################################
#os.chdir("P:\\My Documents\\Library")
#from aod_monthly_mean import aod_monthly_mean

def get_aeronet_aod_550nm_monthly_mean(filename):
    file = Dataset(filename)        
    time = nc.num2date(file.variables['time'][:], file.variables['time'].units)
    aod500 = file.variables['AOD_500nm'][:]
    aod675 = file.variables['AOD_675nm'][:]
    
    alpha = (np.log(aod500/aod675)/np.log(500/675))*-1
    aod550 = aod500*np.power((550/500),-alpha)
    #Calculate monthly mean
    startyr = str(time[0]); startyr=int(startyr[0:4])
    endyr = str(time[-1]); endyr=int(endyr[0:4])
    aod550, time = aod_monthly_mean(aod550,time,startyr,endyr)
    #for ti in range(0,np.size(time)):
    #    time[ti] = toYearFraction(time[ti])
    lat = file.variables['lat'][:]
    lon = file.variables['lon'][:]
    
    return aod550, time, lat, lon

###############################################################################
def get_aeronet_aod_550nm(filename):
    file = Dataset(filename)        
    time = nc.num2date(file.variables['time'][:], file.variables['time'].units)
    aod500 = file.variables['AOD_500nm'][:]
    aod675 = file.variables['AOD_675nm'][:]
    
    alpha = (np.log(aod500/aod675)/np.log(500/675))*-1
    aod550 = aod500*np.power((550/500),-alpha)
    for ti in range(0,np.size(time)):
        time[ti] = toYearFraction(time[ti])
    lat = file.variables['lat'][:]
    lon = file.variables['lon'][:]
    
    return aod550, time, lat, lon

###############################################################################
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

   
def resample_2d(array, sample_pts, query_pts): #*******************************
   # from my_functions import nan_helper
    ''' Resamples 2D array to be sampled along queried points.
    Args:
        array (numpy.ndarray): 2D array.

        sample_pts (tuple): pair of numpy.ndarray objects that contain the x and y sample locations,
            each array should be 1D.

        query_pts (tuple): points to interpolate onto, also 1D for each array.

    Returns:
        numpy.ndarray.  array resampled onto query_pts via bivariate spline.

    '''
    
    xq, yq = np.meshgrid(*query_pts)
    interpf = interpolate.RectBivariateSpline(*sample_pts, array)
    tmp=interpf.ev(xq, yq)  # evaluate algorithm
    return tmp.T 

def nan_helper(y): #**********************************************************
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

################################################################################
def my_interpolate(lat1,lon1,data1,lat2,lon2):
    from my_functions import resample_2d
    from my_functions import nan_helper

    sample_pts = (lat1[:],lon1)
    query_pts = (lat2[:],lon2)
    
    if data1.ndim==2:
        data2 = resample_2d(data1[::-1,:], sample_pts, query_pts)
    else:#assume 3 dimensions
        if np.isnan(data1).any() == True:
        #Use the NaN helper!
            s = np.shape(data1) #assume 3 dimensions
            nansbegone = np.empty((s[0],s[1],s[2])); nansbegone[:,:,:] = np.nan
            for a in range(0,s[0]):
                for b in range(0,s[2]):
                    y=data1[a,:,b]
                    nans, x = nan_helper(y)
                    y[nans] = np.interp(x(nans),x(~nans),y[~nans])
                    nansbegone[a,:,b] = y
            data1 = nansbegone
        
             
        ntime = np.size(data1,0)
        data2 = np.empty((ntime,np.size(lat2),np.size(lon2))) 
        data2[:,:,:] = np.nan
        for ti in range(0,ntime):
            data2[ti,:,:] = resample_2d(data1[ti,::-1,:], sample_pts, query_pts)
        
    output = data2[:,::-1,:]
    return output

################################################################################
def my_interpolate_4d(lat1,lon1,data1,lat2,lon2):
    from my_functions import resample_2d
    from my_functions import nan_helper

    sample_pts = (lat1[:],lon1)
    query_pts = (lat2[:],lon2)
    
    if data1.ndim==2:
        data2 = resample_2d(data1[::-1,:], sample_pts, query_pts)
    if data1.ndim==3:
        if np.isnan(data1).any() == True:
        #Use the NaN helper!
            s = np.shape(data1) #assume 3 dimensions
            nansbegone = np.empty((s[0],s[1],s[2])); nansbegone[:,:,:] = np.nan
            for a in range(0,s[0]):
                for b in range(0,s[2]):
                    y=data1[a,:,b]
                    nans, x = nan_helper(y)
                    y[nans] = np.interp(x(nans),x(~nans),y[~nans])
                    nansbegone[a,:,b] = y
            data1 = nansbegone
        
             
        ntime = np.size(data1,0)
        data2 = np.empty((ntime,np.size(lat2),np.size(lon2))) 
        data2[:,:,:] = np.nan
        for ti in range(0,ntime):
            data2[ti,:,:] = resample_2d(data1[ti,::-1,:], sample_pts, query_pts)
        output = data2[:,::-1,:]

    if data1.ndim==4:
        if np.isnan(data1).any() == True:
        #Use the NaN helper!
            s = np.shape(data1) #assume 3 dimensions
            nansbegone = np.empty((s[0],s[1],s[2],s[3])); nansbegone[:,:,:,:] = np.nan
            for a in range(0,s[0]):
                for b in range(0,s[1]):
                    for c in range(0,s[3]):
                        y=data1[a,b,:,c]
                        nans, x = nan_helper(y)
                        y[nans] = np.interp(x(nans),x(~nans),y[~nans])
                        nansbegone[a,b,:,c] = y
            data1 = nansbegone      
             
        ntime = np.size(data1,1)
        nens = np.size(data1,0)

        data2 = np.empty((nens,ntime,np.size(lat2),np.size(lon2))) 
        data2[:,:,:,:] = np.nan
        for en in range(0,nens):
            for ti in range(0,ntime):
                data2[en,ti,:,:] = resample_2d(data1[en,ti,::-1,:], sample_pts, query_pts)
        output = data2[:,:,::-1,:]
                
    if data1.ndim==5:                
        if np.isnan(data1).any() == True:
        #Use the NaN helper!
            s = np.shape(data1) #assume 5 dimensions
            nansbegone = np.empty((s[0],s[1],s[2],s[3],s[4])); nansbegone[:,:,:,:] = np.nan
            for a in range(0,s[0]):
                for b in range(0,s[1]):
                    for c in range(0,s[2]):
                        for d in range(0,s[4]):
                            y=data1[a,b,c,:,d]
                            nans, x = nan_helper(y)
                            y[nans] = np.interp(x(nans),x(~nans),y[~nans])
                            nansbegone[a,b,c,:,d] = y
            data1 = nansbegone      
        
        ntimes= np.size(data1,2)             
        ntime = np.size(data1,1)
        nens = np.size(data1,0)

        data2 = np.empty((nens,ntime,ntimes,np.size(lat2),np.size(lon2))) 
        data2[:,:,:,:] = np.nan
        for en in range(0,nens):
            for ti in range(0,ntime):
                for tis in range(0,ntimes):
                    data2[en,ti,tis,:,:] = resample_2d(data1[en,ti,tis,::-1,:], sample_pts, query_pts)

        output = data2[:,:,:,::-1,:]
    return output
###############################################################################
def get_seasonal_var(data):
    #Calculate DJF, MAM, JJA, SON from a climatology
    #Assumes 3 dimensions
    djf = np.mean([data[11,:,:],data[0,:,:],data[1,:,:]],0)
    mam = np.mean([data[2,:,:],data[3,:,:],data[4,:,:]],0)
    jja = np.mean([data[5,:,:],data[6,:,:],data[7,:,:]],0)
    son = np.mean([data[8,:,:],data[9,:,:],data[10,:,:]],0)
    data_out = np.stack((djf,mam,jja,son),0)
    
    return data_out

###############################################################################
def find_index(array, point_of_interest):
    idx = (np.abs(array-point_of_interest)).argmin()
    return idx

###############################################################################
def calc_trop_o3_column(var,lev):
    
    #For a single timestep, this function calculates tropospheric column ozone abundance in DU. Here the tropospheric column
    #is defined as the surface to the 250 hPa level. 
    #"Var" should be in mole/mole, and have dimensions [lev,lat,lon]. 
    #"lev" should be in hPa.
    
    #Converted the function from Matlab to Python on 3 Aug 2018, Laura Revell, UC.

    #Set up constants:
    g = 9.81# gravity, m/s2
    avagadro = 6.02252e23 # molec/mol
    mdry = 0.028964 # molec. wt. of dry air, kg/mol
    const = 0.01 * avagadro / (g * mdry)
    du = 2.69e16 #factor to convert molecules/cm2 to DU

    ptp = 250 #defined here as the tropopause height, hPa
    surf = 1000 #surface, hPa
    
    #Integrate vertically:
    dim2 = np.size(var,1)
    dim3 = np.size(var,2)

    tropos_col = np.empty((dim2,dim3)); tropos_col[:,:] = np.nan
    
    for la in range(0,dim2):
        for lo in range(0,dim3):
            
            idx1 = find_index(lev,ptp)
            idx2 = find_index(lev,surf)
            
            var_working = var[:,la,lo]
            lev_working = lev
            
            #check for NaNs/masked values
            lev_working = lev_working[~np.isnan(var_working)] #Update lev
            var_working = var_working[~np.isnan(var_working)] #Removes NaNs
            lev_working = lev_working[~var_working.mask] #Update lev
            var_working = var_working[~var_working.mask] #Removes masked NaNs

            #Construct pressure and concentration variables from surface to 250 hPa
            #Check that pressure runs from surface upwards, i.e. lev[0] = 1000 hPa
            if idx1<idx2:
                var_working = var_working[::-1]
                lev_working = lev_working[::-1]
                idx1=np.size(lev)-idx1
            
            press_var = lev_working[:idx1]
            conc_var = var_working[:idx1]
            
            #Calculate tropospheric columns
            col=0
            for i in range(1,np.size(press_var)):
                term = 0.5*(conc_var[i] + conc_var[i-1]) * np.abs(press_var[i-1] - press_var[i]) * const
                col = col + term
            
            tropos_col[la,lo] = col/du 
            del press_var; del conc_var
     
    x = tropos_col
    return x

###############################################################################
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

###############################################################################
def polar_stereographic_plot_contourf(fig,lat,lon,data,upperlat,clevs,cmap,no_contours,cbar_label,pole):
    m = Basemap(projection=pole,boundinglat=upperlat,lon_0=90,resolution='l')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,10.))
    m.drawmeridians(np.arange(-180.,181.,30.))
    
    x=np.zeros((np.size(lat),np.size(lon)),np.float64)
    y=np.zeros((np.size(lat),np.size(lon)),np.float64)
    for la in range(0,np.size(lat)):
        for lo in range(0,np.size(lon)):
            x[la,lo],y[la,lo] = m(lon[lo],lat[la])

    cs = m.contourf(x,y,data,cmap=cmap,levels=np.linspace(np.min(clevs),np.max(clevs),no_contours))
    cs.set_clim(np.min(clevs),np.max(clevs))
    cbar = fig.colorbar(cs, ticks=[clevs])
    cbar.set_label(cbar_label,rotation=90)
    return cs

###############################################################################
def polar_stereographic_plot_contourf_and_contour(fig,i,lat,lon,data,upperlat,vmin,vmax,clev,cmap,lat2,lon2,data2,no_contours):
    m = Basemap(projection='spstere',boundinglat=upperlat,lon_0=90,resolution='l')
    m.drawcoastlines()
    #m.drawparallels(np.arange(-80.,81.,10.))
    #m.drawmeridians(np.arange(-180.,181.,30.))
    
    x=np.zeros((np.size(lat),np.size(lon)),np.float64)
    y=np.zeros((np.size(lat),np.size(lon)),np.float64)
    for la in range(0,np.size(lat)):
        for lo in range(0,np.size(lon)):
            x[la,lo],y[la,lo] = m(lon[lo],lat[la])

    cax = m.contourf(x,y,data,cmap=cmap,levels=np.linspace(vmin,vmax,no_contours))
    cax.set_clim(vmin,vmax)
    
    x2=np.zeros((np.size(lat2),np.size(lon2)),np.float64)
    y2=np.zeros((np.size(lat2),np.size(lon2)),np.float64)
    for la in range(0,np.size(lat2)):
        for lo in range(0,np.size(lon2)):
            x2[la,lo],y2[la,lo] = m(lon2[lo],lat2[la])
            
    cs2 = m.contour(x2, y2, data2, no_contours, colors='k')
    plt.clabel(cs2, fontsize=8, inline=1, fmt='%.2f')
    
    if i==0:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(cax, cax=cbar_ax, ticks=[clev])
        cbar.set_label('AOD at 550 nm', rotation=90)

    return cax,cs2

###############################################################################
def calc_density(temp,pres):
    #define constants:
    R = 287  #universal gas constant for dry air in J/kg/K
    density = pres/(R*temp)
    
    return density
    
###############################################################################
def check_lat_lon(lat,lon,array):
    from my_functions import t_lon
    from my_functions import t_lon_array
    #ensure lat runs N-S and lon runs 180W-180E
    if lat[0]>1:
        lat = lat[::-1]
        array = array[::-1,:]
    if lon[-1]>180:
        array = t_lon_array(lon,array)
        lon = t_lon(lon)
        
    return lat, lon, array
def check_lat_lon_model(lat,lon,array):
    #ensure lat runs N-S and lon runs 180W-180E
#     if lat[0]<1:
#         lat = lat[::-1]
#         array = array[:,::-1,:]
    if lon[-1]<180:
        array = t_lon_array(lon,array)
        lon = t_lon_360(lon)
        
    return lat, lon, array
###############################################################################
def check_lev_lat_lon(lat,lon,array):
    #like check_lat_lon but ignores level dimension in 2nd position
    from my_functions import t_lon
    from my_functions import t_lon_array
    #ensure lat runs N-S and lon runs 180W-180E
    if lat[0]>1:
        lat = lat[::-1]
        array = array[:,:,::-1]
    if lon[-1]>180:
        array = t_lon_array(lon,array)
        lon = t_lon(lon)
        
    return lat, lon, array
def check_lev_lat_lon_model(lat,lon,array):
    #like check_lat_lon but ignores level dimension in 2nd position
    from my_functions import t_lon_360
    from my_functions import t_lon_array
    #ensure lat runs N-S and lon runs 180W-180E
#     if lat[0]<1:
#         lat = lat[::-1]
#         array = array[:,:,::-1,:]
    if lon[-1]<180:
        array = t_lon_array(lon,array)
        lon = t_lon_360(lon)        
    return lat, lon, array
###############################################################################
def make_seasonal_mean(array):
    #assumes an array of shape time, lat, lon and converts it to an array of time, lat, lon where the time dimension
    #is 4 seasonal means, i.e. DJF, MAM, JJA, SON.
    s = np.shape(array)
    array_seasonal = np.empty((4,s[1],s[2])); array_seasonal[:,:,:] = np.nan
    array_seasonal[0,:,:] = np.nanmean(np.stack((array[11],array[0],array[1]),2),2)
    array_seasonal[1,:,:] = np.nanmean(np.stack((array[3],array[4],array[5]),2),2)
    array_seasonal[2,:,:] = np.nanmean(np.stack((array[6],array[7],array[8]),2),2)
    array_seasonal[3,:,:] = np.nanmean(np.stack((array[9],array[10],array[11]),2),2)
    
    return array_seasonal


        
def Normalization_nan(data0,data1):
    """ 
    the data should be flattened containing
    all what is wanted to use to compare. 
    data0 = a dataset used (such as atmos DMS)
    data1 = a different kind of data showing connected to data 1 (such as oceanic DMS) 
    """
    TAN_nan= data0[np.logical_not(np.isnan(data1))]
    TAN_MODIS_nan= data1[np.logical_not(np.isnan(data1))]
    TAN_nan=np.where(np.isnan(TAN_nan),0,TAN_nan)

    norm_dms=Normalization(TAN_nan)
    norm_flux=Normalization(TAN_MODIS_nan)
    r2=np.corrcoef(norm_flux,norm_dms)[0][1]**2
    return r2


def calc_pdf(data, left, right, nbins):
    '''
    This is a function to calculate pdf
data: data to calculate PDF must be 1D
left: left bound of data
right: right bound of data
    nbins: number of bins
    '''
    data = data[~np.isnan(data)]
    right = right-0.1
    
    bins = np.linspace(left, right, nbins)
    
    frequency = np.zeros(len(bins))
    
    for i in range(0,nbins):
        if np.isnan(np.nanmean(data)) == False:
            if i < nbins-1:
                mask = (bins[i]<= data) & (data < bins[i+1])
                frequency[i] = len(data[mask])/len(data)
            else:
                mask = data >= bins[i]
                frequency[i] = len(data[mask])/len(data)
        else:
            frequency[i] = np.nan
    return(bins,frequency)

def Normalization(data):
    mean = np.mean(data)
    normalized_data = [x - mean for x in data]
    std = np.std(normalized_data)
    normalized_data = [x / std for x in normalized_data]
    return normalized_data

def calc_cdf(perd):
    cpd = np.zeros(len(perd))
    for i in range(0,len(perd)):
        cpd[i] = perd[i]+ cpd[i-1]
    return(cpd)

# Function to create map plots
def plot_map_Basemap(ax, data, xa, ya, cmap, levels=None, title=None):
    os.chdir('/home/ybhatti/prjs1076/Emulator')

    map_ax = Basemap(llcrnrlon=0., llcrnrlat=-85, urcrnrlon=360., urcrnrlat=85,
                     projection='cyl', resolution='c', ax=ax)
    x, y = map_ax(xa, ya)
    map_ax.drawcoastlines()
    map_ax.drawcountries()
    map_ax.drawmeridians(np.arange(0., 360, 60.), labels=[0, 0, 0, 1], fontsize=8)
    map_ax.drawparallels(np.arange(-90., 90, 30.), labels=[1, 0, 0, 0], fontsize=8)
    if levels is not None:
        cs = map_ax.contourf(x, y, data, levels, extend='both', cmap=cmap)
    else:
        cs = map_ax.contourf(x, y, data, extend='both', cmap=cmap)
    if title:
        ax.set_title(title, fontsize=14)
    return cs


import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from typing import Optional, Dict, Any
from matplotlib.contour import QuadContourSet

from matplotlib.colors import LogNorm

from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm

def plot_map(
    ax: plt.Axes, 
    data: np.ndarray, 
    lons: np.ndarray, 
    lats: np.ndarray, 
    cmap: str, 
    levels: Optional[np.ndarray] = None, 
    title: Optional[str] = None,
    title_fontsize: int = 12,  # Renamed and set a default fontsize for the title
    log_scale: bool = False,  # New parameter to switch to log scale,
    gridline_kwargs: Optional[Dict[str, Any]] = None
) -> QuadContourSet:
    """
    Plots a filled contour map on a specified axis.
    
    Parameters:
        ax (plt.Axes): The matplotlib axis to plot on.
        data (np.ndarray): The data to plot.
        lons (np.ndarray): Longitudes corresponding to the data.
        lats (np.ndarray): Latitudes corresponding to the data.
        cmap (str): Colormap to use for plotting.
        levels (Optional[np.ndarray]): Contour levels; if None, defaults to the colormap levels.
        title (Optional[str]): Title for the plot.
        log_scale (bool): If True, applies a logarithmic color scale.
        gridline_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for gridlines.
    
    Returns:
        QuadContourSet: The contour set created by the plot.
    """
    # Set projection to PlateCarree for simple lat/lon grids
   # projection = ccrs.PlateCarree()

    # Add cyclic point to avoid discontinuities at the 180° meridian
    data, lons = add_cyclic_point(data, coord=lons)

    # Plot map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Default gridline settings
    default_gridline_kwargs = {
        'draw_labels': True,
        'linewidth': 0.5,
        'color': 'gray',
        'alpha': 0.5,
        'linestyle': '--',
        'xlabel_style': {'size': 7},  # Default label font size
        'ylabel_style': {'size': 7}   # Default label font size
    }

    # Update default settings with any user-provided gridline kwargs
    if gridline_kwargs is not None:
        default_gridline_kwargs.update(gridline_kwargs)

    gl = ax.gridlines(**default_gridline_kwargs)  # Apply gridline settings

    # Use LogNorm for logarithmic scaling if log_scale is True
    norm = LogNorm() if log_scale else None
 #   norm = BoundaryNorm(boundaries=clevs,extend='both')
    # Plot the contour or filled contour
    cs = ax.contourf(lons, lats, data, levels=levels, norm=norm, cmap=cmap, extend='both')

    if title:
        ax.set_title(title, fontsize=title_fontsize)  # Set title font size

    return cs
#new_samples = pd.DataFrame(0, index=range(n), columns=ppe_param.columns)
# Extract the PPE_Control row as a Series

def PPE_Uncertainty(n,VARIABLE,ppe_control_values):    
        #data = new_samples
    # Create a new DataFrame with `n` rows, where each row is filled with the `PPE_Control` values
    new_samples = pd.DataFrame([ppe_control_values] * n, index=range(n))

    new_samples[VARIABLE] = np.random.uniform(0, 1, n)  # e.g. V_SCALE_EMI_SS with values between 0.5 and 2.5

    m, sd = gp_model_.batch_stats(new_samples, batch_size=n)
    mnn, sdnn = nn_model.batch_stats(new_samples, batch_size=n)
    print(f"Memory usage: {get_memory_usage()} GB")
    
    print(f"Memory usage: {get_memory_usage()} GB")
    gc.collect()
    return m ,sd, mnn, sdnn

def Total_PPE_Uncertainty(n,ppe_control_values):    
        #data = new_samples
    # Create a new DataFrame with `n` rows, where each row is filled with the `PPE_Control` values
    new_samples = pd.DataFrame([ppe_control_values] * n, index=range(n))

    new_samples[:] = np.random.rand(*new_samples.shape)

    m, sd = gp_model_.batch_stats(new_samples, batch_size=n)
    mnn, sdnn = nn_model.batch_stats(new_samples, batch_size=n)
    print(f"Memory usage: {get_memory_usage()} GB")
    
    print(f"Memory usage: {get_memory_usage()} GB")
    gc.collect()
    return m ,sd, mnn, sdnn

# n=10000
# new_samples = pd.DataFrame([ppe_control_values] * n, index=range(n))

# new_samples[:] = np.random.rand(*new_samples.shape)
# gp_samples, _ = gp_model_.predict(new_samples.values)
# cnn_samples, _ = nn_model.predict(new_samples.values)
# dots=find_diffs(cnn_samples, gp_samples,0)

## Creating index for PPE to be used for uncertainy calculation

import copy
def parameter_testing(table,PARAMETER):
    para_table=copy.deepcopy(table)
    # Assuming 'Control' row is the first row (index 0)
    control_row = para_table.iloc[0]
    
    # Update all rows to match the Control row, except for V_SCALE_CDNC_MIN
    for column in table.columns:
        if column != PARAMETER:
            para_table[column] = control_row[column]
    return para_table
