import ast
import numpy as np
import pandas as pd
from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity


#pd.set_option('display.max_columns', 5)

class Preprocessor():
    _parent = "SpotGenTrack"
    _children = {"Sources" : "Data Sources",
                 "Features" : "Features Extracted",
                 "Preprocessed":"Preprocessed Datasets"
                 }
    _trackspath = f"{_parent}/{_children}"
    
    @staticmethod    
    def LoadCSV(filename:str='spotify_tracks')->pd.DataFrame:

        csv_path = FindCSVFile(Preprocessor._parent,filename)
        if not csv_path:
            print(f"CSV file '{filename}.csv' not found.")
            return pd.DataFrame() 

        return pd.read_csv(csv_path)
    
    
    @staticmethod
    def Artists()->pd.DataFrame:
        artists = Preprocessor.LoadCSV('spotify_artists')
        #drop unecessary column(s)
        artists = artists.drop(['Unnamed: 0','track_name_prev', 'type'], axis=1)  
        artists = artists.rename(columns = {'id':'artists_id','name':'artists_name'})
        return artists
    
    @staticmethod
    def Albums()->pd.DataFrame:
        albums = Preprocessor.LoadCSV('spotify_albums')
        albums = albums.drop(['Unnamed: 0','available_markets', 'external_urls', 'href', 'images', 'uri', 'type'], axis=1) 
        albums = albums.rename(columns = {'id':'album_id'})
        albums.loc[albums['release_date_precision']=='year','release_date'] +='-01-01'
        albums.loc[albums['release_date_precision']=='month','release_date'] +='-01'
        albums["release_date"] = pd.to_datetime(albums["release_date"])
        albums = albums.drop(['release_date_precision'], axis=1)
        return albums
    
    @staticmethod
    def Tracks()->pd.DataFrame:
        tracks = Preprocessor.LoadCSV('spotify_tracks')
        tracks = tracks[tracks['popularity'] != 0.0]
        popularity_threshold = tracks['popularity'].quantile(0.1)
        tracks = tracks[tracks['popularity'] > popularity_threshold]
        tracks = tracks.drop_duplicates(subset=['name'])
        tracks.reset_index(drop=True, inplace=True)
        
        tracks['artists_id'] = tracks['artists_id'].apply(ast.literal_eval)
        tracks['artists_id'] = tracks['artists_id'].str[0]

        temp = pd.DataFrame(data={'track_id':tracks.id,'name':tracks.name})
        temp.to_csv(f"{Preprocessor._children['Preprocessed']}/tracknames_map.csv",index=None)
        tracks = tracks.drop(['Unnamed: 0','analysis_url', 'disc_number', 'available_markets', 'href', 'mode', 'lyrics',\
            'preview_url', 'track_href', 'track_name_prev','type','track_number','name','uri'], axis=1)
        tracks = tracks.rename(columns = {'id':'track_id', 'popularity': 'track_popularity'})

        tracks['country'] = pd.factorize(tracks['country'])[0]  #label encoding

        """
        print(tracks['duration_ms'].describe())->
        count    1.019390e+05
        mean     2.467708e+05
        std      1.904303e+05
        min      1.155000e+03
        25%      1.840000e+05
        50%      2.168930e+05
        75%      2.610550e+05
        max      5.505831e+06
        """
        t1 = Preprocessor.LoadCSV('spotify_albums')
        t1 = t1.rename(columns={'artist_id':'artists_id'})
        
        t2 = Preprocessor.LoadCSV('spotify_artists')
        t2 = t2.rename(columns={'id':'artists_id','name':'artists_name'})
        temp = pd.merge(t1[['track_id','artists_id']],
                t2[['artists_id','artists_name','genres']],
                on='artists_id',
                how='left')
        
        temp = temp.dropna()
        temp = temp.loc[temp['genres']!='[]']
        
        tracks = pd.merge(tracks, (enc:=extract_genres(temp)),on='track_id')
        #standardization of duration_ms
        tracks['duration_ms'] = MinMaxScaler().fit_transform(np.array(tracks['duration_ms']).reshape(-1,1))
        return tracks
    
    @staticmethod
    def Lyrics()->pd.DataFrame:
        lyrics = Preprocessor.LoadCSV('lyrics_features')
        lyrics = lyrics.drop(['Unnamed: 0'], axis=1)
        
        return lyrics
    
    @staticmethod
    def Merger()->pd.DataFrame:
        tracks = Preprocessor.Tracks()
        artists = Preprocessor.Artists()
        lyrics = Preprocessor.Lyrics()
        extended_df = pd.merge(tracks,artists[['artists_id', 'artist_popularity']], on='artists_id')
        
        numeric_df = extended_df.select_dtypes(include=['number'])
        corr=numeric_df.corr().round(2)
        yield "artists_tracks",corr
        
        extended_df = extended_df.drop(['loudness'], axis=1)
        
        tal_df = pd.merge(extended_df, lyrics, on='track_id') #tal -> Track Artists Lyrics
        
        numeric_df = tal_df.select_dtypes(include=['number'])
        corr=numeric_df.corr().round(2)
        yield "artists_tracks_lyrics",corr
        
        tal_df = tal_df.drop(['mean_syllables_word'], axis=1)
        
        numeric_df = tal_df.select_dtypes(include=['number'])
        corr=numeric_df.corr().round(2)
        yield "merged_data", corr
        
        tal_df.to_csv(f"{Preprocessor._children['Preprocessed']}/raw_dataset.csv",index=None)
        
        f_cols = ['album_id','artists_id','track_id','playlist','artists_name']
        for col in f_cols:
            tal_df[col], mapped_col = pd.factorize(tal_df[col])

            mapping_df = pd.DataFrame({
                'track_id': mapped_col.values
            })
            mapping_df['encoded'] = mapping_df.index
            mapping_df.to_csv(f"{Preprocessor._children['Preprocessed']}/{col}_map.csv",index=None)
        
        tal_df.to_csv(f"{Preprocessor._children['Preprocessed']}/dataset.csv",index=None)
        
        
    def GenerateCosine()->None:
        #if os.path.exists(f"{Preprocessor._children['Preprocessed']}/cosine.npz"):
            #return f"np.load(f'{Preprocessor._children['Preprocessed']}/cosine.npz')"
        for name, corr in Preprocessor.Merger():
            pass
        dataset = pd.read_csv(f"{Preprocessor._children['Preprocessed']}/dataset.csv")
        names = pd.read_csv(f"{Preprocessor._children['Preprocessed']}/tracknames_map.csv")
        track_ids = pd.read_csv(f"{Preprocessor._children['Preprocessed']}/track_id_map.csv")
        
        conn = pd.merge(names,track_ids,on='track_id')
        
        conn.rename(columns={"track_id": "original_track_id", "encoded": "track_id"},inplace=True)
        dataset = pd.merge(dataset, conn[['track_id','name']],on='track_id')
        dataset.rename(columns={"track_id":"encoded","original_track_id":"track_id"},inplace=True)
        
        
        dataset = dataset.drop_duplicates(subset=['encoded'])
        temp = dataset['encoded'].to_numpy()
        songnames = dataset["name"].to_numpy()
        dataset.drop(columns=['encoded','name','album_id','artists_id','time_signature','artists_name'],axis=1,inplace=True)
        dataset.to_csv(f"{Preprocessor._children['Preprocessed']}/preprocessed_dataset.csv")
        print("Generating cosine using columns: ", dataset.columns)
        
        cosine = cosine_similarity(dataset)
        indices = pd.DataFrame(data={'encoded':temp,'name':songnames})
        indices.to_csv(f"{Preprocessor._children['Preprocessed']}/indices.csv")
        print("Saving indices.")
        #return f"np.load(f'{Preprocessor._children['Preprocessed']}/cosine.npz')"#dont save just send to main
        return cosine
    
def main():
    pass
    pre = Preprocessor()
    pre.Merger()
    
if __name__=='__main__':
    main()