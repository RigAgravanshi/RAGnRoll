import pandas as pd
from src.config_loader import CFG

df = pd.read_csv(CFG['data']['raw_path'])

def bucket_audio_features(df) -> pd.DataFrame :

    ''' Removing these features for the following reasons:
    loudness: badly distributed. Will use for Later-Phase Filtering
    liveness: no one searches songs for their "liveness", they would rather go for energy/danceability/vibing. Bad distribution.
    speechiness: overlaps with instrumentalness, and itself badly distributed
    '''
    
    df['energy_label'] = pd.cut(df['energy'],
                                 bins=[0, 0.33, 0.66, 1.0],
                                 labels=['low energy', 'medium energy', 'high energy'],
                               include_lowest=True)

    df['danceability_label'] = pd.cut(df['danceability'],
                                  bins=[0, 0.4, 0.7, 1.0],
                                  labels=['low danceability', 'danceable', 'high danceability'],
                                     include_lowest=True)
    
    df['instrumentalness_label'] = pd.cut(df['instrumentalness'],
                                  bins=[0, 0.3, 0.7,1.0],
                                  labels=["vocal-heavy", "mixed vocals", "heavy-instrumental"],
                                         include_lowest=True)   
    
    df['acousticness_label'] = pd.cut(df['acousticness'],
                                  bins=[0, 0.33, 0.66, 1.0],
                                  labels=["electric", "semi-acoustic", "acoustic"],
                                     include_lowest=True) 
    
    df['valence_label'] = pd.cut(df['valence'],
                                  bins=[0, 0.33, 0.66, 1.0],
                                  labels=['melancholic', 'neutral', 'upbeat'],
                                include_lowest=True)    
    
    df['tempo_label'] = pd.cut(df['tempo'],
                                  bins=[0, 90, 140, 250],
                                  labels=["slow", "moderate", "fast"],
                              include_lowest=True)    
  
    df['duration_label'] = pd.cut(df['duration_ms'],
                                  bins=[0, 150000, 360000, 2100000],    # <2.5 min short song, >6 min long song
                                  labels=["short", "moderate length", "long"],
                                 include_lowest=True)    

    df['popularity_label'] = pd.cut(df['popularity'],                    # considering the distribution of data and observational-analysis
                                  bins=[0, 50, 90, 100],
                                  labels=["hidden-gem", "known", "popular"],
                                   include_lowest=True) 
    
    return df

def song_description(row):
    return (
        f"{row['track_name']} by {row['artist_name']}. Genre: {row['genre']}. "
        f"{row['energy_label']}, {row['danceability_label']}, {row['instrumentalness_label']}. "
        f"{row['acousticness_label']}, {row['valence_label']}, {row['tempo_label']}. "
        f"{row['duration_label']}. {row['popularity_label']}."
    )

df_label = bucket_audio_features(df)
df_label['song_text'] = df_label.apply(song_description, axis=1)

# All Label_Data has been put into song_text column.
# dropping the label columns

label_cols = ['energy_label', 'danceability_label', 'instrumentalness_label',
            'acousticness_label', 'valence_label', 'tempo_label',
            'duration_label', 'popularity_label']

df_processed = df_label.drop(columns=label_cols)

print(df_processed.shape)

df_processed.to_csv(CFG['data']['processed_path'], index=False)