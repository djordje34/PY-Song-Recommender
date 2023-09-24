import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from preprocessor import Preprocessor

class Recommender():
    

    
    def __init__(self) -> None:
        self._indices = pd.read_csv(f"{Preprocessor._children['Preprocessed']}/indices.csv")
        self._indices.index = self._indices['name']  
        self._indices = self._indices.drop(['name'],axis=1)
        self._original_id = pd.read_csv(f"{Preprocessor._children['Preprocessed']}/track_id_map.csv")
        self._songnames = pd.read_csv(f"{Preprocessor._children['Preprocessed']}/tracknames_map.csv")
        
        self._songnames = pd.merge(self._songnames,self._original_id,on='track_id')
        artists = Preprocessor.LoadCSV('spotify_artists')
        artists.drop(['Unnamed: 0'],axis=1, inplace=True)
        artists = artists.rename(columns={'name':'artist_name'})
        self._songnames = pd.merge(self._songnames,artists[['artist_name','genres','track_id','artist_popularity']],on='track_id',how='left')
        self._songnames.dropna(subset=['genres'],inplace=True)
        self._songnames.encoded=self._songnames.encoded.astype(int)
        self._songnames.drop_duplicates(subset=['encoded'],inplace=True)
        self._songnames.sort_values(by=['artist_popularity'],ascending=False)
        
        print("Recommender initialized.")
    
    def fit(self)->None:
        self._cosine = Preprocessor.GenerateCosine()
        print("Fitting done.")
        
    def predict(self,song_title:str=None):
        score=list(enumerate(self._cosine[self._indices.loc[self._indices.index== song_title,'encoded'].values[0]]))
        similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
        
        similarity_score = similarity_score[1:151]        
        top_songs_index = [i[0] for i in similarity_score]
        top_songs= self._songnames[self._songnames['encoded'].isin(top_songs_index)]
        top_songs.sort_values(by=['artist_popularity'],ascending=False, inplace=True)
        print("Prediction done.")
        return top_songs.drop(['encoded'],axis=1)
    
    
    def __str__(self) -> str:
        return f"Recommender based on cosine similarity specifically for content based recommendations of songs."



def main(): #make sure that the 
    rec = Recommender()
    rec.fit()
    ls = rec.predict('Going Bad (feat. Drake)')
    print(ls)
    
if __name__=='__main__':
    main()