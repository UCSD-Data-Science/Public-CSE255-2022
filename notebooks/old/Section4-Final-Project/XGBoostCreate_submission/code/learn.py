from lib.KDTreeEncoding import *

import xgboost as xgb
from lib.XGBHelper import *
from lib.XGBoost_params import *
from lib.score_analysis import *

from lib.logger import logger

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from glob import glob
import pandas as pd
import pickle as pkl
import sys
from time import time
import os

class timer:
    def __init__(self):
        self.t0=time()
        self.ts=[]
    def mark(self,message):
        self.ts.append((time()-self.t0,message))
        print('%6.2f %s'%self.ts[-1])
    def _print(self):
        for i in range(len(self.ts)):
            print('%6.2f %s'%self.ts[i])

def train_boosted_trees(D):
    ### Train and test
    # set parameters for XGBoost
    param['max_depth']=2
    param['num_round']=10

    ### Train on random split, urban and rural together

    train_selector=np.random.rand(df.shape[0]) > 0.7
    Train=D.get_subset(train_selector)
    Test=D.get_subset(~train_selector)

    param['num_round']=10
    log10=simple_bootstrap(Train,Test,param,ensemble_size=30)
    param['num_round']=100
    log100=simple_bootstrap(Train,Test,param,ensemble_size=30)

    styled_logs=[
        {   'log':log10,
            'style':['k:','k-'],
            'label':'10 iterations',
            'label_color':'k'
        },
        {   'log':log100,
            'style':['r:','r-'],
            'label':'100 iterations',
            'label_color':'r'
        }
    ]
    return styled_logs

if __name__=='__main__':
    poverty_dir=sys.argv[1]
    T=timer()
    depth=8   #for KDTree

    ## load file list
    image_dir=poverty_dir+'/anon_images'


    files=glob(f'{image_dir}/*.npz')
    print(f'found {len(files)} files')

    T.mark('listed files')
    train_table='../public_tables/train.csv'
    df=pd.read_csv(train_table,index_col=0)
    df.index=df['filename']

    ## Generate encoding tree
    train_size,tree=train_encoder(files,max_images=500,tree_depth=depth)
    T.mark('generated encoder tree')
    ## Encode all data using encoding tree
    Enc_data=encoded_dataset(image_dir,df,tree,label_col='label')
    T.mark('encoded images')
    D=DataSplitter(Enc_data.data)
    styled_logs=train_boosted_trees(D)

    _mean,_std=plot_scores(styled_logs,title='All')
    T.mark('trained trees')
    
    os.makedirs('data', exist_ok=True)
    pickle_file='data/Checkpoint.pk'
    Dump={'styled_logs':styled_logs,
         'tree':tree,
         'mean':_mean,
         'std':_std}
    pkl.dump(Dump,open(pickle_file,'wb'))
    T.mark('generated pickle file')
    print('picklefile=',pickle_file)
    T._print()
