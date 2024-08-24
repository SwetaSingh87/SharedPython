#!/usr/bin/env python
# coding: utf-8

# # Importing Header

# In[2]:


#%run Header.ipynb


# In[3]:


# import import_ipynb 
# from Header import *


# In[4]:


from Header import *


# # 1. Data Preprocessing (Spectra)

# ## Smoothing Filter ( First Order)
# 

# In[7]:


# Input spectra as features

def sgsmooth (spectrum,window):
    filt_spec = spectrum.iloc[:,:].copy()
    (row,col) = spectrum.shape
    if row == 1:
        print('you should have given spectra as a dataframe -- not series')
    if window != 0:
        for c in range (0,col):
            if c>= window and c <= (col-window):      ## SAFE
                filt_spec.iloc[:,c] = spectrum.iloc[:,(c-window):(c+window)].mean(axis=1).copy()
            elif c < window and c <= (col-window):    ##  LEFT
                filt_spec.iloc[:,c] = spectrum.iloc[:,0:(c+window)].mean(axis=1).copy()
            elif c > (col-window) and c >= window:    ## RIGHT
                filt_spec.iloc[:,c] = spectrum.iloc[:,(c-window):col].mean(axis=1).copy()
            else:     ## LEFT & RIGHT  c < window and c > (col-window)
                filt_spec.iloc[:,c] = spectrum.iloc[:, 0:col].mean(axis=1).copy()        
    return (filt_spec.iloc[:,:].copy())


# ## Smoothing Filter ( Savitzky Golay First and Second Order)

# In[9]:


# -------------- Smoothing Spectra using sg1: (savgol order 1) and sg2: (savgol order 2)  -----------

#--INPUT: window_len (should be odd positive integer), filt_type (should be 'sg1' or 'sg2')


def filt_sg(spectra, window_len, filt_type):
    sg = filt_type
    w = window_len
    
    if sg == 'sg1':
        if w ==0 or w == 1:
            smth_spec = spectra.copy()   
        else:
            smth_spec = spectra.copy()
            pd.DataFrame(savgol_filter(smth_spec, w, 1, axis=1), columns=smth_spec.columns, index=smth_spec.index)
            
    else:
        if w ==0 or w == 1:
            smth_spec = spectra.copy()   
        else:
            smth_spec = spectra.copy()
            pd.DataFrame(savgol_filter(smth_spec, w, 2, axis=1), columns=smth_spec.columns, index=smth_spec.index)
            
    return (smth_spec)

#--OUTPUT: smoothed spectra with same column names and row indices as original (input) spectra


# ## First Order Derivative

# In[11]:


def fod (spectra):
    fo_spec = spectra.iloc[:,:].copy()
    (row,col) = fo_spec.shape
    
    for i in range(0,col):
        if i==col-1:
            fo_spec.iloc[:,i] = fo_spec.iloc[:,i-1]
        else:    
            fo_spec.iloc[:,i] = (spectra.iloc[:,i+1]- spectra.iloc[:,i])
        
    #fo_spec = 100*fo_spec
    return(fo_spec.copy())


# ## Continuum Removal

# In[13]:


def continuum_removal(points, show=False):
    x1, y1 = points.T
    augmented = np.concatenate([points, [(x1[0], np.min(y1)-1), (x1[-1], np.min(y1)-1)]], axis=0)
    hull = ConvexHull(augmented)
    continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
    continuum_function = interp1d(*continuum_points.T)
    yprime = continuum_function(x1) - y1
    #yprime = y1 / continuum_function(x1)

    if show:
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(x1, y1, label='Data')
        axes[0].plot(*continuum_points.T, label='Continuum')
        axes[0].legend()
        axes[1].plot(x1, yprime, label='Data / Continuum')
        axes[1].legend()

    return yprime


def continuum_removed(spectra):
    cr_spec = spectra.copy()    
    row, col = spectra.shape
    x1 = np.arange (0, col, 1)
    
    
    for r in range(0,row,1):
        y1 = cr_spec.iloc[r,:]
        points = np.c_[x1, y1]
        yprime = continuum_removal(points, show=False)
        cr_spec.iloc[r,:] = yprime
        
    return cr_spec



# ## Resampling (n_bands)

# In[15]:


def resample_spectra (spectra, n_band):
    row, width = spectra.shape
    if n_band == 0:
        red_spectra = spectra.copy()
    else:
        w = width/n_band
        
        #----- obtaining the sampling locations in indx----
        indx = []
        for i in range (0,n_band,1):
            indx.append(np.floor((i+0.5)*w))
            
        #------ applying smoothing filter on spectra---------
        temp_smooth = sgsmooth (spectra, np.floor(0.5*w).astype(int))
        
        #------ picking values at sampling locations---------
        red_spectra = temp_smooth.iloc[:, indx].copy()
        
    return (red_spectra)


# In[16]:


#------- adaptive binning of 'spectra' in 'n_band' using different 'score' for each wavelengths-----
def ad_resample_spectra (spectra, n_band, score):
    row, width = spectra.shape
    
#     std_score = 10*(spectra.std()/max(spectra.std()))
    
#     r_val, p_val = find_rpval (spectra, T[0])
#     print('r_val: ', r_val.iloc[0,:])
#     r_val.replace(np.nan, 0, inplace=True)
#     cor_score = abs(10*r_val.iloc[0,:].copy())
#     corr_score = 10*(cor_score/max(cor_score))
    
#     wave_score = std_score + corr_score
    
    wave_score = score
    
    print(wave_score)
    
    sum_score = wave_score.sum()
    band_score = sum_score/n_band
    print('sum_score:', sum_score, '----> n_band:',n_band, '-----> band_score', band_score)     
    
    if n_band == 0:
        red_spectra = spectra.copy()
    else:
        #----- finding boundaries of bins--------------
        cur_score = 0
        prev_score = 0
        boundary = [0]
        print('initial boundary', boundary)
        for j in range (0, width):
            l = len(boundary)
            cur_score = prev_score + wave_score.iloc[j] 
            print('current boundary at j=',j, boundary)
            print('cur_score:', cur_score, 'l*band_score:', l*band_score, 'prev_score:', prev_score)
            if (cur_score >= (l*band_score) and prev_score < (l*band_score)):
                boundary.append(j)
                print('updated boundary')
                     
            prev_score = cur_score
            
        boundary.append(width-1)
        print('final boundary', boundary)
        
        #---- obtaining the sampling locations (mid points) in indx----
        indx = []
        for i in range (0,n_band,1):
            print('i+1 is:', i+1)
            indx.append(int(0.5*(boundary[i]+ boundary[i+1])))
        
        #---- initializing the output (resampled) spectra----
        red_spectra = spectra.iloc[:, indx].copy()
        
        
        #---- finalizing the value at the sampled locations---
        for c in range (0, len(indx)):
            min_c = boundary[c]
            max_c = boundary[c+1]
            red_spectra.iloc[:, c] = spectra.iloc[:, min_c:max_c].mean(axis=1).copy()            
       
    return  red_spectra 


# # 2. Data Preprocessing (Target Variables)

# ## Z-score Normalization 

# In[19]:


def z_score (target):
    X = target.copy()
    # outliers removal
    mean = X.mean()
    std = X.std()
    X.loc[abs((X - mean)) >= 4*std] = mean
    
    X = (X-mean)/std
    return (X.copy())
 

# udf = pd.read_csv('uae.csv')
# Y = lognormal (udf['TOC'].copy())
# plt.hist(udf['TOC']/2.5, bins=98)
# plt.show()
# plt.hist(Y, bins=98)
# plt.show()



# ## Standard Normalization (Log+MinMax)

# In[21]:


def min_max_normal (target):
    X = target.copy()
    # outliers removal
    mean = X.mean()
    std = X.std()
    X.loc[abs((X - mean)) >= 4*std] = mean
    # shift min to 1
    m = X.min()
    dis = (1-m)
    X = X + dis    
    # apply log transform
    #X = np.log(X)    
    # normalize/rescale using Min-Max method 
    minX = X.min()
    maxX = X.max()
    diff = maxX - minX
    X = ((X-minX)*10)/diff    
    return (X.copy())


# ## Calculating Pearson Correlation Coefficient

# In[23]:


# Pearson corelation between different wavelengths and Targets/Outputs (i.e, sand, clay, silt, and TOC) 

def find_rpval (spectra, tar):
    (r, c) = spectra.shape
    
    r_val = spectra.iloc[[0], :].copy()
    p_val = spectra.iloc[[0], :].copy()
    
    for j in range(0, c):
        r_val.iloc[0,j], p_val.iloc[0,j] = stats.pearsonr(tar, spectra.iloc[:, j])
    
    return(r_val, p_val)


# # 3. Train Test Split

# In[25]:


# best_split finds random state and minimum error for best train test split

def best_split (X,y,tst_siz):
    ymin = y.min()
    ymax = y.max()
    trn_siz = 1-tst_siz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= tst_siz, random_state=0)
    rand_st = 0
    bin_train = np.histogram(y_train, bins = 8, range = (ymin,ymax), density=False)
    bin_test = np.histogram(y_test, bins = 8, range = (ymin,ymax), density=False)
    error = abs((bin_train[0])/trn_siz - (bin_test[0])/tst_siz)
    cum_error = error.sum()
    min_err= cum_error
    
    for i in np.arange(1,42,1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= tst_siz, random_state=i)
        bin_train = np.histogram(y_train, bins = 8, range = (ymin,ymax), density=False)
        bin_test = np.histogram(y_test, bins = 8, range = (ymin,ymax), density=False)
        error = abs((bin_train[0])/trn_siz - (bin_test[0])/tst_siz)
        cum_error = error.sum()
        if cum_error < min_err:
            min_err = cum_error
            rand_st = i
            #print(i)

    return (rand_st, min_err)


# In[26]:


# # ## Plotting the Distribution of Train and Test Output Data 
# plt.style.use(['science','notebook','grid'])

# fig, ax = plt.subplots(2,2, figsize=(18,14))

# minSand = np.min(y_sand)
# maxSand = np.max(y_sand)

# binsSand = np.linspace(minSand, maxSand, 8)
# # density =True : used to normalise bin heights to make the integral of  histogram 1.
# ax[0][0].hist([ySand_train, ySand_test], binsSand , label=['Train', 'Test'], density=True, color = ['blue','red'])
# # results in error when yN_train/ yN_test is data frame or ndarray
# ax[0][0].legend(loc='upper left', fontsize =12)
# ax[0][0].set_xlabel('Sand content',fontsize =16)
# ax[0][0].set_ylabel('Normalised frequency',fontsize =12)
# ax[0][0].tick_params(axis='both', labelsize=8)

# minSilt = np.min(y_silt)
# maxSilt = np.max(y_silt)

# binsSilt = np.linspace(minSilt, maxSilt, 8)
# # density =True : used to normalise bin heights to make the integral of  histogram 1.
# ax[0][1].hist([ySilt_train, ySilt_test], binsSilt , label=['Train', 'Test'], density=True, color = ['blue','red'])
# # results in error when yN_train/ yN_test is data frame or ndarray
# ax[0][1].legend(loc='upper right', fontsize =12)
# ax[0][1].set_xlabel('Silt content',fontsize =16)
# ax[0][1].set_ylabel('Normalised frequency',fontsize =12)
# ax[0][1].tick_params(axis='both', labelsize=8)

# minClay = np.min(y_clay)
# maxClay = np.max(y_clay)

# binsClay = np.linspace(minClay, maxClay, 8)
# # density =True : used to normalise bin heights to make the integral of  histogram 1.
# ax[1][1].hist([yClay_train, yClay_test], binsClay , label=['Train', 'Test'], density=True, color = ['blue','red'])
# # results in error when yN_train/ yN_test is data frame or ndarray
# ax[1][1].legend(loc='upper right', fontsize =12)
# ax[1][1].set_xlabel('Clay content',fontsize =16)
# ax[1][1].set_ylabel('Normalised frequency',fontsize =12)
# ax[1][1].tick_params(axis='both', labelsize=8)

# minTOC = np.min(y_toc)
# maxTOC = np.max(y_toc)

# binsTOC = np.linspace(minTOC, maxTOC, 8)
# # density =True : used to normalise bin heights to make the integral of  histogram 1.
# ax[1][0].hist([yTOC_train, yTOC_test], binsTOC , label=['Train', 'Test'], density=True, color = ['blue','red'])
# # results in error when yN_train/ yN_test is data frame or ndarray
# ax[1][0].legend(loc='upper right', fontsize =12)
# ax[1][0].set_xlabel('Total Organic Content',fontsize =16)
# ax[1][0].set_ylabel('Normalised frequency',fontsize =12)
# ax[1][0].tick_params(axis='both', labelsize=8)

# fig.suptitle('Train Test Distribution of Data', x = 0.5 ,y = .95, fontsize = 20)

# #plt.show()


# # 4. IQRP, RPD, R2, RMSE

# In[28]:


def find_iqrp (Yp, Y):
    mse = mean_squared_error(Yp, Y)
    rmse = np.sqrt(mse)
    X = Y.copy()
    l = len(X)
    q1 = math.floor(l/4)
    q3 = math.floor(3*l/4)
    X = X.sort_values().reset_index(drop=True)
    res = (X[q3] - X[q1])/rmse
    return(res)

def find_rpd (Yp, Y):
    mse = mean_squared_error(Yp, Y)
    rmse = np.sqrt(mse)
    res = Y.std()/rmse
    return(res)

def find_r2 (Yp, Y):
    res = r2_score(Y, Yp)
    return(res)

def find_rmse(Yp, Y):
    res = np.sqrt(mean_squared_error(Y, Yp))
    return(res)

    


# # Parameters for Best Model Fit 

# ## PLSR

# In[31]:


def best_param_PLSR (X, y, rand_n, max_n_comp):       
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=rand_n)
    iqrpL = []
    
    for n in range(1,max_n_comp):
        Model = PLSRegression(n_components=n, scale=True)
        Model.fit(X_train, y_train)
        y_pred = Model.predict(X_test, copy=True)        
        iqrp_test = find_iqrp(y_pred, y_test)
        iqrpL.append(iqrp_test)                
    
    IQRP = max(iqrpL)
    n_iqrp = iqrpL.index(max(iqrpL))+1
    
    #print('IQRP :', IQRP,  '>>> n_comp: ', n_iqrp)    
    return (n_iqrp)
        


# In[ ]:





# In[ ]:




