import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import pickle
#Combine all the spatial-temporal covariates over the US, and then augment with the temporal ones matching that date
#Default is 1985-2019 as before 1985 we have no soil moisture data
def combine_covariates(year_range=range(1985,2019), data_path='/share/data/willett-group/'):
    
    mei = pd.read_hdf(data_path + '/mei.h5')

    mjo_phase = pd.read_hdf(data_path + '/mjo_phase_updated.h5')
    mjo_phase = pd.DataFrame(mjo_phase)
    mjo_phase.columns = ['mjo_phase']
    mjo_phase.index.name = 'start_date'

    mjo_amp = pd.read_hdf(data_path + '/mjo_amplitude_updated.h5')
    mjo_amp = pd.DataFrame(mjo_amp)
    mjo_amp.columns = ['mjo_amp']
    mjo_amp.index.name = 'start_date'

    nao = pd.read_hdf(data_path + '/nao.h5')
    dfs = [mjo_phase, mjo_amp, nao]
    temporal_dat = mei
    for df_ in dfs:
        temporal_dat = temporal_dat.merge(df_, on=['start_date'])

    for year in year_range:
        tmp2m = pd.read_hdf(data_path + '/tmp2m.'+str(year)+'.h5')
        sm = pd.read_hdf(data_path + '/sm.'+str(year)+'.h5') #Empty until 1985?
        slp = pd.read_hdf(data_path + '/slp.'+str(year)+'.h5')
        rhum500 = pd.read_hdf(data_path + '/rhum500.'+str(year)+'.h5')
        rhumsig995 = pd.read_hdf(data_path + '/rhum.sig995.'+str(year)+'.h5')
        precip = pd.read_hdf(data_path + '/precip.'+str(year)+'.h5')
        icec = pd.read_hdf(data_path + '/icec.'+str(year)+'.h5')
        hgt700 = pd.read_hdf(data_path + '/hgt700.'+str(year)+'.h5')
        hgt500 = pd.read_hdf(data_path + '/hgt500.'+str(year)+'.h5')
        hgt200 = pd.read_hdf(data_path + '/hgt200.'+str(year)+'.h5')
        hgt10 = pd.read_hdf(data_path + '/hgt10.'+str(year)+'.h5')
        dfs = [sm, slp, rhum500, rhumsig995, precip, hgt700, hgt500, hgt200, hgt10]
        df_names = ['sm', 'slp', 'rhum500', 'rhumsig995', 'precip', 'hgt700', 'hgt500', 'hgt200', 'hgt10']
        df = tmp2m
        for df_, df_name in zip(dfs, df_names):
            if(isinstance(df_, pd.Series)):
                df_ = df_.rename(df_name)
            df = df.merge(df_, on=['lat','lon','start_date'])
        df = df.join(temporal_dat)
        df.to_hdf(data_path + '/covariates_all_'+str(year)+'.h5', key='covariates')

# Normalize any dataframe according to specified metric, only need to give avgperiod if doing avg     
def normalize(data, method='zscore', avgperiod=14):
    if method == 'zscore':
        if type(data) is np.ndarray:
            return zscore(data, axis=0)
        else:
            return data.apply(zscore)
    elif method == 'log':
        x_min = data.min()
        transformed = data - x_min + 1
        return transformed.apply(lambda col: np.log10(col) if np.issubdtype(col.dtype, np.number) else col)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    elif method == 'avg':
        return data.rolling(window=avgperiod).mean().dropna()
    
#Turn our dataframes into numpy arrays for much faster operations, like transposing
#This func takes about 2 or 3 min per year, so don't worry if it seems like its doing nothing
def convert_covariates_to_numpy(year_range=range(1985, 2019), data_path='/share/data/willett-group/'):
    for y in year_range:
        year = pd.read_hdf(data_path + 'covariates_all_'+str(y)+'.h5')
        a = np.zeros((366 if y % 4 ==0 else 365,3274,10))
        indexs = np.array([np.array(i) for i in year.index.to_numpy()])

        days = {day : idx for idx, day in enumerate(np.unique(indexs[:,2]))}
        locs = {loc : idx for idx, loc in enumerate(np.unique(np.array(list(map(lambda loc: str(loc[0]) + '_' + str(loc[1]), indexs[:,0:2])))))}


        for idx, row in year.iterrows():
            i = days[idx[2]]
            j = locs['' + str(idx[0]) + '_' + str(idx[1])]
            a[i,j] = row.to_numpy()[0:10]
        days_inv = {v: k for k, v in days.items()}
        locs_inv = {v: k for k, v in locs.items()}
        year_data = {'data':a, 'days':days_inv, 'locs':locs_inv}
        pickle.dump(year_data, open(data_path + 'covariates_as_numpy_'+str(y)+'.pkl', "wb"))

        
#Each day has 169727 locations of SSTs, we need to turn that into a 2D array for PC

def convert_ssts_to_numpy(year_range = range(1985, 2019), data_path='/share/data/willett-group/'):
    for y in year_range:
        year = pd.read_hdf('/scratch/grosenthal/sst.'+str(y)+'.h5')
        a = np.zeros((366 if y % 4 ==0 else 365,169727))

        days = {day : idx for idx, day in enumerate(year.index.unique(level='start_date'))}
        locs = {loc : idx for idx, loc in enumerate(np.unique(np.array(list(map(lambda loc: str(loc[0]) + '_' + str(loc[1]), year.reset_index(level=['start_date']).index.unique())))))}


        for idx, row in year.iteritems():
            i = days[idx[2]]
            j = locs['' + str(idx[0]) + '_' + str(idx[1])]
            a[i,j] = row
        days_inv = {v: k for k, v in days.items()}
        locs_inv = {v: k for k, v in locs.items()}
        year_data = {'data':a, 'days':days_inv, 'locs':locs_inv}
        pickle.dump(year_data, open(data_path + 'sst_as_numpy_'+str(y)+'.pkl', "wb"))
        
def concatenate_ssts(year_range = range(1985,2019), data_path='/share/data/willett_group/'):
    all_ssts = []
    for i in range(1985,2019):
        year = pickle.load( open( "/share/data/willett-group/sst_as_numpy_"+str(i)+'.pkl', "rb" ) )
        all_ssts.append(year['data'])
    all_ssts_concat = np.concatenate(all_ssts, axis=0)
    all_sst = {'data':all_ssts_concat, 'locs': year['locs']} 
    pickle.dump(all_sst, open(data_path + 'all_sst.pkl', "wb"))
    
    
# Flatten each year into a 365x32470 array and then concatenate them and then return a dataframe with them
# Must be done before feeding into PCA
def transpose_and_concatenate(year_range=range(1985, 2019), data_path='/share/data/willett-group/'):
    transposed_arrs = []
    days_index = []
    for i in year_range:
        year_data = pickle.load( open( data_path + 'covariates_as_numpy_'+str(i)+'.pkl', "rb" ) )
        transposed = np.array([row.reshape(-1) for row in year_data['data'][:,:, 0:10]])
        transposed_arrs.append(transposed)
        days_index = days_index + list(year_data['days'].values())
    concatenated = np.concatenate(transposed_arrs, axis=0)
    
    covars = ['tmp2m', 'sm', 'slp', 'rhum500', 'rhumsig995', 'precip', 'hgt700', 'hgt500', 'hgt200', 'hgt10']
    
    #Since the year_data['locs'] are ordered the same way as they appear in year_data, then we can just use them as labels
    columns = [loc + '_' + c for loc in year_data['locs'].values() for c in covars]
    days_index.insert(2768, pd.Timestamp('1992-07-31 00:00:00'))
    return pd.DataFrame(concatenated, columns=columns, index=days_index)

def LandPCA(year_range = range(1985, 2019), data_path='/share/data/willett-group/'):
    all_years = transpose_and_concatenate(year_range, data_path)
    all_years = normalize(all_years)
    
    train_years = all_years[0:int(all_years.shape[0] * 0.8)]
    test_years = all_years[int(all_years.shape[0] * 0.8):]
    
    pca = PCA()
    X_train_pca = pca.fit_transform(train_years)
    
    X_test_pca = pca.transform(test_years)
    
    pickle.dump(pca, open(data_path + 'land_pca_mapping.pkl', "wb"))
    pickle.dump(X_train_pca, open(data_path + 'land_train_pca.pkl', "wb"))    
    pickle.dump(X_test_pca, open(data_path + 'land_test_pca.pkl', "wb"))
    
def SSTPCA(year_range = range(1985, 2019), data_path='/share/data/willett-group/'):
    sst = pickle.load( open( data_path + "all_sst.pkl", "rb" ) )
    sst = normalize(sst['data'])
    
    train_sst = sst[0:int(sst.shape[0] * 0.8)]
    test_sst = sst[int(sst.shape[0] * 0.8):]
    
    pca = PCA()
    sst_train_pca = pca.fit_transform(train_sst)
    
    sst_test_pca = pca.transform(test_sst)
    
    pickle.dump(pca, open(data_path + 'sst_pca_mapping.pkl', "wb"))
    pickle.dump(sst_train_pca, open(data_path + 'sst_train_pca.pkl', "wb"))    
    pickle.dump(sst_test_pca, open(data_path + 'sst_test_pca.pkl', "wb"))        

def detrend(all_y, year_range = range(1985, 2019)):
    year_lengths = [366 if i % 4 == 0 else 365 for i in year_range]
    
    
    train_days = int(0.8*sum(year_lengths))
    test_days = sum(year_lengths) - int(0.8*sum(year_lengths))
    
    year_indices = np.cumsum(year_lengths)
    
    train_indices = year_indices[year_indices <= train_days]
    test_indices = year_indices[year_indices >= train_days]
    
    train_indices = np.append(train_indices, train_days)
    test_indices = np.insert(test_indices, 0, train_indices[-1])

    test_indices[-1] -= 28
    
    split_by_year = np.array_split(all_y[:train_days], train_indices)
    
    test_split_by_year = np.array_split(all_y[train_days:], (test_indices - train_days)[1:])[:-1]
    
    temps_by_location_by_year = np.zeros((3274, 366, 34))
    for year in range(len(split_by_year)):
        for day in range(len(split_by_year[year])):
            for loc in range(len(split_by_year[year][day])):
                temps_by_location_by_year[loc, day, year] = split_by_year[year][day, loc]
    
    test_temps_by_location_by_year = np.zeros((3274, 366, 34))
    for year in range(len(test_split_by_year)):
        for day in range(len(test_split_by_year[year])):
            for loc in range(len(test_split_by_year[year][day])):
                
                real_day = year_lengths[year] + 1 - len(test_split_by_year[year]) + day if not year else day
#                 if not year:
#                     print('original day:' + str(day) + ', realday: ' + str(real_day))
                test_temps_by_location_by_year[loc, real_day, year] = test_split_by_year[year][day, loc]
    
    means = np.zeros((3274, 366))
    stds = np.zeros((3274, 366))
    
    
    
    from scipy import stats
    for loc in range(len(temps_by_location_by_year)):
        for day in range(len(temps_by_location_by_year[loc])):
            before_replace = temps_by_location_by_year[loc, day]
            before_replace[before_replace == 0] = np.nan
            means[loc, day] = np.nanmean(before_replace)
            stds[loc, day] = np.nanstd(before_replace)
            
            temps_by_location_by_year[loc,day] = stats.zscore(before_replace, nan_policy='omit')
            
            
    for loc in range(len(test_temps_by_location_by_year)):
        for day in range(len(test_temps_by_location_by_year[loc])):
            before_replace = test_temps_by_location_by_year[loc, day]
            before_replace[before_replace == 0] = np.nan
            
            test_temps_by_location_by_year[loc,day] = (test_temps_by_location_by_year[loc,day] - means[loc, day])/stds[loc,day]
            
    flipped = np.zeros((3274, 34, 366))

    flipped_test = np.zeros((3274, 34, 366))
    
    for loc in range(len(temps_by_location_by_year)):
        for y in range(0, 34):
            flipped[loc, y] = temps_by_location_by_year[loc, :, y]
            
            
    for loc in range(len(test_temps_by_location_by_year)):
        for y in range(0, 34):
            flipped_test[loc, y] = test_temps_by_location_by_year[loc, :, y]
                 
    all_days = np.zeros((3274, len(all_y)))
    for loc in range(len(flipped)):
        testloc = np.concatenate((np.concatenate(flipped[loc]), np.concatenate(flipped_test[loc])))
        all_days[loc] = testloc[~np.isnan(testloc)]

    return all_days.T, (means, stds)    

def create_sequence(input_data, y_actual, train_window, flatten_sequence):  #  Create sequences of observations for training
    seq = []
    L = len(input_data)
    feature_size = input_data[0].shape[0]
    for i in range(L-train_window):
        train_seq = np.array(input_data[i:i+train_window])
        if flatten_sequence:
            train_seq = np.reshape(train_seq, (-1, ))
        train_label = np.array(y_actual[i + train_window - 1]) # get y_t+14
        seq.append((train_seq ,train_label))
    return seq

def create_dataset_detrend( year_range = range(1985, 2019), n_comp = 100, n_sst_comp = 100, train_window=90, flatten_sequence=True, detrend_y=True, target_variable='tmp2m', average_period = 14, data_path = '/share/data/willett-group/'):
    
    train_normal = pickle.load( open( data_path + "land_train_pca.pkl", "rb" ) )[:,0:n_comp]
    train_sst = pickle.load( open( data_path + "sst_train_pca.pkl", "rb" ) )[0:train_normal.shape[0], 0:n_sst_comp]   
    
    train_data = np.concatenate((train_normal,train_sst), axis=1)
    
    
    test_normal = pickle.load( open( data_path + "land_test_pca.pkl", "rb" ) )[:-28,0:n_comp]
    print(test_normal.shape)
    test_sst = pickle.load( open( data_path + "sst_test_pca.pkl", "rb" ) )[0:test_normal.shape[0], 0:n_sst_comp]   
    
    test_data = np.concatenate((test_normal,test_sst), axis=1)
    
    all_x = np.concatenate((train_data, test_data))
   
    all_years = transpose_and_concatenate(range(1985, 2019), data_path)
    target_vars = list(filter(lambda c: target_variable in c, all_years.columns))
#     Forward looking rolling mean is really difficult, so I reverse all the columns, shift them forward by 14, calculate 14 day average, and then reverse them again
    all_y = all_years[target_vars][::-1].shift(13).rolling(window=14).mean()[::-1][:-28].to_numpy()
    all_y_detrended = []
    means = []
    stds = []
    if(detrend_y):
        all_y_detrended, (means, stds) = detrend(all_y) 
    
    else:
        all_y_detrended = all_y
    all_seq = create_sequence(all_x, all_y_detrended,train_window, flatten_sequence)
    
    train = all_seq[0:train_sst.shape[0]]
    test = all_seq[train_sst.shape[0]-90:]
    
   
    return (train, test), (means, stds)

def dataset_to_pairs(train, test):
    train_X = []
    train_Y = []
    
    test_X = []
    test_Y = []

    for train_pair in train:
        train_X.append(train_pair[0])
        train_Y.append(train_pair[1])      
        
    for test_pair in test:
        test_X.append(test_pair[0])
        test_Y.append(test_pair[1])
        
    return (np.array(train_X), np.array(train_Y)), (np.array(test_X), np.array(test_Y))     

def create_dataset( year_range = range(1985, 2019), n_comp = 100, n_sst_comp = 100, target_variable='tmp2m', average_period = 14, data_path = '/share/data/willett-group/'):
    
    train_normal = pickle.load( open( data_path + "land_train_pca.pkl", "rb" ) )[:,0:n_comp]
    train_sst = pickle.load( open( data_path + "sst_train_pca.pkl", "rb" ) )[0:train_normal.shape[0], 0:n_sst_comp]   
    
    train_data = np.concatenate((train_normal,train_sst), axis=1)
    
    
    test_normal = pickle.load( open( data_path + "land_test_pca.pkl", "rb" ) )[:-28,0:n_comp]
    test_sst = pickle.load( open( data_path + "sst_test_pca.pkl", "rb" ) )[0:test_normal.shape[0], 0:n_sst_comp]   
    
    test_data = np.concatenate((test_normal,test_sst), axis=1)
    
            
    all_years = transpose_and_concatenate(range(1985, 2019), data_path)
    target_vars = list(filter(lambda c: target_variable in c, all_years.columns))
#     Forward looking rolling mean is really difficult, so I reverse all the columns, shift them forward by 14, calculate 14 day average, and then reverse them again
    all_y = all_years[target_vars][::-1].shift(13).rolling(window=14).mean()[::-1]
    train_y = all_y[0:train_sst.shape[0]].to_numpy()
    test_y = all_y[train_sst.shape[0]:-28].to_numpy()
    
   
    return (train_data, train_y), (test_data, test_y)
