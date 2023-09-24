from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessor import Preprocessor



class Analyzer():
    _parent = 'plots'
    tracks = Preprocessor.Tracks()
    albums = Preprocessor.Albums()
    artists = Preprocessor.Artists()
    lyrics = Preprocessor.Lyrics()
    colors = sns.color_palette("rocket")
    colors_cmap = sns.color_palette("rocket",as_cmap=True)
    @staticmethod
    def CountryPiePlot()->None:
        labels = ['Belgium', 'Finland', 'Argentina']
        values = Analyzer.tracks.country.value_counts()/Analyzer.tracks.country.shape[0]
        plt.pie(values, labels = labels, colors = Analyzer.colors[3:],autopct='%.0f%%', normalize=False,textprops={'color':'black'})
        plt.title("Country Distribution")
        plt.savefig(f"{Analyzer._parent}/country_pie_plot.png")
        
    @staticmethod
    def ArtistsRelevancePlot()->None:
        sns.scatterplot(x=Analyzer.artists.followers, y=Analyzer.artists.artist_popularity, color=Analyzer.colors[2], alpha=0.8)
        plt.xlabel('Followers Count')
        plt.ylabel('Popularity')
        plt.title('Popularity vs. Followers Count')
        plt.grid(True)
        plt.savefig(f"{Analyzer._parent}/artists_followers_vs_popularity.png")


    @staticmethod
    def AlbumsPlot()->None:
        Analyzer.albums['year'] = Analyzer.albums['release_date'].dt.year
        albums_by_year = Analyzer.albums.groupby('year').size()
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=albums_by_year.index, y=albums_by_year.values,palette="flare")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=90)
        plt.xlabel('Year')
        plt.ylabel('Number of Albums')
        plt.title('Number of Albums per Year')
        plt.savefig(f"{Analyzer._parent}/number_of_albums_per_year.png")
        
    @staticmethod
    def TrackKeyPopularityPlot()->None:
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        
        key_dataframe = pd.DataFrame({'key': Analyzer.tracks.key, 'track_popularity_mean': Analyzer.tracks.track_popularity})
        key_dataframe.groupby(['key'], as_index=True).aggregate({'track_popularity_mean': 'mean'})
        
        energy_dataframe = pd.DataFrame({'energy':Analyzer.tracks.energy, 'track_popularity_mean': Analyzer.tracks.track_popularity})
        energy_dataframe['energy_level'] = pd.cut(energy_dataframe['energy'], bins=bins)
        energy_dataframe.drop(['energy'],axis=1)
        energy_dataframe = energy_dataframe.groupby('energy_level', as_index=True)['track_popularity_mean'].mean()
        
        dance_dataframe = pd.DataFrame({'danceability':Analyzer.tracks.danceability, 'track_popularity_mean': Analyzer.tracks.track_popularity})
        dance_dataframe['danceability_level'] = pd.cut(dance_dataframe['danceability'], bins=bins)
        dance_dataframe.drop(['danceability'],axis=1)
        dance_dataframe = dance_dataframe.groupby('danceability_level', as_index=True)['track_popularity_mean'].mean()

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)
        
        ax0 = fig.add_subplot(gs[0, :])
        sns.barplot(x=key_dataframe.key, y=key_dataframe.track_popularity_mean, palette="rocket", ax=ax0)
        ax0.set_xlabel('Key')
        ax0.set_ylabel('Popularity (mean)')
        ax0.set_title('Popularity of Each Track Key')

        ax1 = fig.add_subplot(gs[1, 0])
        sns.barplot(x=energy_dataframe.index, y=energy_dataframe.values, palette="rocket", ax=ax1)
        ax1.set_xlabel('Energy Level')
        ax1.set_ylabel('Popularity (mean)')
        ax1.set_title('Popularity of Each Energy Level')

        ax2 = fig.add_subplot(gs[1, 1])
        sns.barplot(x=dance_dataframe.index, y=dance_dataframe.values, palette="rocket", ax=ax2)
        ax2.set_xlabel('Danceability Level')
        ax2.set_ylabel('Popularity (mean)')
        ax2.set_title('Popularity of Each Danceability Level')
        
        plt.tight_layout()
        plt.savefig(f"{Analyzer._parent}/popularity_of_parameters.png")
        
    @staticmethod
    def CorrelationMatrix()->None:
        for name, corr in Preprocessor.Merger():
            sns.set(font_scale=0.85)
            plt.figure(figsize=(8,8))
            sns.set_palette("rocket")
            sns.set_style("white")
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(corr,annot=True,cmap='rocket_r',mask=mask,cbar=True, square=True)
            plt.title('Correlation matrix')
            plt.tight_layout()
            plt.savefig(f"{Analyzer._parent}/correlation_matrix_{name}.png")

def main():
    pass
    #Analyzer.CountryPiePlot()
    #Analyzer.ArtistsRelevancePlot()
    #Analyzer.AlbumsPlot()
    #Analyzer.TrackKeyPopularityPlot()
    #Analyzer.CorrelationMatrix()
    
if __name__=='__main__':
    main()