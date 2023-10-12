import pickle
import pandas as pd

df = pd.read_csv('data/highered.csv')

df.to_pickle('data/highered.pkl') 