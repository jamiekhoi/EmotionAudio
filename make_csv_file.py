import argparse
import os
import pandas as pd

IDENTIFIERS = ['Modality',
               'VocalChannel',
               'Emotion',
               'EmotionalIntensity',
               'Statement',
               'Repitition',
               'Actor']

EMOTIONS = {
    '01': 'Neutral',
    '02': 'Calm',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fearful',
    '07': 'Disgust',
    '08': 'Surprised'}


def get_id(id):
    return IDENTIFIERS.index(id)


def create_ravdess_csv(kind='Speech'):
    df = pd.DataFrame(columns=['FilePath', 'Modality', 'VocalChannel', 'Emotion',
                               'EmotionalIntensity', 'Statement', 'Repitition', 'Actor', 'Gender'])
    dataset_path = 'Datasets/Audio_{}_Actors_01-24'.format(kind)

    actors = os.listdir(dataset_path)
    for actor in actors:
        actor_files = os.listdir(os.path.join(dataset_path, actor))
        for file in actor_files:
            ids = file[:-4].split('-')
            filedict = {}
            filedict['FilePath'] = os.path.join(dataset_path, actor, file)
            filedict['Filename'] = file
            for ID in IDENTIFIERS:
                if ID == 'Emotion':
                    filedict[ID] = EMOTIONS[ids[get_id(ID)]]
                    continue
                elif ID == 'Actor':
                    filedict['Gender'] = 'Male' if int(ids[get_id(ID)]) % 2 else 'Female'
                filedict[ID] = ids[get_id(ID)]
            df = df.append(filedict, ignore_index=True)
    print(df)
    if not os.path.exists('Processed/RAVDESS{}'.format(kind.lower())):
        os.makedirs('Processed/RAVDESS{}'.format(kind.lower()))
    df.to_csv('Processed/RAVDESS{}/ravdess{}.csv'.format(kind.lower(), kind), index=False)


def main():
    """
    Use to create csv file with information about data to use later
    :return:
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument('--makecsv', action='store_true')
    parser.add_argument('--dataset', required=True, choices=['ravdesssong', 'ravdessspeech'])  # , default=os.path.expanduser('~/data'))
    args = parser.parse_args()

    if args.dataset == 'ravdesssong':
        print('Making Ravdess Song')
        create_ravdess_csv(kind='Song')
    elif args.dataset == 'ravdessspeech':
        print('Making Ravdess Speech')
        create_ravdess_csv(kind='Speech')
    else:
        print('No option')


if __name__ == '__main__':
    main()
