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
        year = pd.read_hdf(data_path + 'sst.'+str(y)+'.h5')
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
        year = pickle.load( open(data_path+'sst_as_numpy_'+str(i)+'.pkl', "rb" ) )
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
