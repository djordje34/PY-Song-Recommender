import os
import pandas as pd

def FindCSVFile(dir_path,filename):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.csv') and file == filename + '.csv':
                        return os.path.join(root, file)
            return None
        
def extract_genres(ar:pd.DataFrame):
    genre_keywords = {
        'pop': ['pop'],
        'rock': ['rock'],
        'hip hop': ['hip hop','urban', 'rap'],
        'rap': ['rap','hip hop','trap'],
        'rock': ['rock'],
        'metal': ['metal'],
        'folk': ['folk'],
        'classical': ['classical'],
        'indie': ['indie'],
        'dnb': ['dnb','drum and bass','jungle'],
        'blues': ['blues'],
        'comedy': ['comedy'],
        'jazz': ['jazz'],
        'reggae': ['reggae'],
        'drill': ['drill','grime'],
        'trap': ['trap','electro','electronic'],
        'orchestra': ['orchestra'],
        'metal': ['metal'],
        'house': ['house']
    }
    
    ar = ar[['track_id','genres','artists_name']]
    for genre,keyw in genre_keywords.items():
        ar[genre] = 0

    for genre,keyw in genre_keywords.items():
        for key in keyw:
            res = ar['genres'].str.contains(key)
            ar.loc[res, genre] += 1

    ar = ar.drop(['genres'],axis = 1)       
    ar = ar[ar['track_id'].str.len() == 22]
    return ar


def extract_artists(ar:pd.DataFrame):
    unique_artists = set()
    for artists_list in ar['artists_id']:
        artists = artists_list.strip("[]").replace("'", "").split(', ')
        unique_artists.update(artists)
    return unique_artists
    
#extract_artists(pd.read_csv('SpotGenTrack/Data Sources/spotify_tracks.csv'))