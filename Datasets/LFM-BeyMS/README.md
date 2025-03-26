# LFM-BeyMS Dataset

This dataset is based on the LFM-1b [1] and the Cultural LFM-1b [2] datasets. LFM-BeyMS includes equally-sized groups of both, beyond-mainstream and mainstream music listeners.

LFM-BeyMS contains
* 4,148 users
* 1,084,922 tracks
* 110,898 artists
* 16,687,363 listening events

## Creation
Beyond-mainstream and mainstream music listeners are found by setting a threshold for the mainstreaminess M-global-R-APC [2]. We obtain beyond-mainstream usergroups by clustering tracks, listened by beyond-mainstream music listeners, based on their acoustic features. Beyond-mainstream music listeners are then assigned to track clusters, which consitutes the user groups. Additionally, genres of tracks are found by matching last.fm tags to Spotify's microgenres. 

### LFM-1b
* LFM-1b_users.txt: User ids of users from the LFM-1b dataset with basic information.
* LFM-1b_users_additional.txt: Additional information about users from the LFM-1b dataset.
* events.tsv: Listening events of users from the LFM-1b dataset. One listening event is given by user id, artist id, album id, track id and unix timestamp.

### Cultural LFM-1b
* acoustic_features_lfm_id.tsv: Spotify's audio description features for a subset of tracks in LFM-1b.
* hofstede.tsv: Hofstede's cultural dimensions.
* world_happiness_report_2018.tsv: World Happiness Report data from 2018.

### Other
* genre_annotations.csv: Genres of tracks.
* user_mainstreaminess.txt: Mainstreaminess measurement of users.

## Dataset
* beyms.csv: User ids of 2,074 beyond-mainstream music listeners
* ms.csv: User ids of 2,074 mainstream music listeners
* events.csv: Listening events of both, beyond-mainstream and mainstream music listeners. One listening event is given by user id, artist id, album id, track id and unix timestamp.
* user_groups.csv: Classifies beyond-mainstream users into four usergroups of unique music taste (with main genres folk, hardrock, ambient, electronica).
* genre_annotations.csv: Genres of tracks listened by beyond-mainstream and mainstream music listeners.
* mainstreaminess.csv: Mainstreaminess values (M_global_R_APC) of beyond-mainstream and mainstream music listeners within this dataset.


## References
[1] Schedl, M. (2016, June). The lfm-1b dataset for music retrieval and recommendation. In Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval (pp. 103-110).
[2] Bauer, C., & Schedl, M. (2019). Global and country-specific mainstreaminess measures: Definitions, analysis, and usage for improving personalized music recommendation systems. PloS one, 14(6).
[3] Eva Zangerle. (2019). Culture-Aware Music Recommendation Dataset [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3477842




