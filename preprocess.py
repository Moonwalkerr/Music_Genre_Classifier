import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import json

DATASET_PATH = '../Data/genres_original/'
JSON_PATH = 'data.json'
SAMPLE_RATE = 22050
DURATION = 30  # measured in secs
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION




# json path where we want to store all the mfccs & labels of the respective sounds
# num_segments = 5, chopping the sound into multiple segments to get more input data for training 

def save_mfcc(dataset_path,json_path, n_mels=13,n_fft=2028,hop_length=512,num_segments=5,SAMPLE_RATE=22050):

#     dictionary to store data
    data = {
        "mapping":[],
        "mfcc":[],
        "label":[],
    }


    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)    
     
#   ensuring we have same num of samples from duration 
    expected_num_mfccs_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)


#     loop through all the genres
#     recursively going throuhout all the dataset
    for i,( dir_path, dir_names, file_names ) in enumerate(os.walk(dataset_path)):

#        ensuring that we are not at the root level, 
#        os.walk gives dataset_path folder at first iteration that we have to skip
        if dir_path is not dataset_path:
        
        
#         save the sematic  labels --> mappings
            dir_path_components = dir_path.split('/')  # genre/blues => [genre, blues]
            semantic_label = dir_path_components[-1]  # considering last component 
            data['mapping'].append(semantic_label)
            print("\nProcessing {}".format(semantic_label ))
            
#        next we will go through all the files in currrent dir_path / genre folder
#         processing files for specific genre
            for f in file_names:

#      load audio file
#      file_names 'f' just gives the file name not path, so extracting path         
                file_path = os.path.join(dir_path,f)
                audio, sr = librosa.load(file_path,sr=SAMPLE_RATE)
                
            
#             dividing the signal to diff segments for more input data
#             extracting the mfccs
#             storing data
                for seg in range(num_segments):
                    start_sample = num_samples_per_segment * seg   
                    # s=0 for first --> s=num_samples_per_seg
                        
                    finish_sample = start_sample + num_samples_per_segment

                    mfccs = librosa.feature.mfcc(audio[start_sample:finish_sample],
                                                 sr=SAMPLE_RATE,
                                                 n_mfcc=n_mels,
                                                 n_fft=n_fft,
                                                hop_length=hop_length)
                    
                    mfccs = mfccs.T

                    
#                    ensuring our samples are of same length per segment
#                     store mfccs per segment             
                    if len(mfccs) == expected_num_mfccs_vectors_per_segment:
                        data['mfcc'].append(mfccs.tolist())
                        data['label'].append(i-1)  
                # i is the counter from the enumerate() method
                # i-1 : ignoring the dataset path i.e 1st iteration output
                        print("\n{}, segment:{}".format(file_path,seg+1))  # seg+1 = to start from 1
                    
        with open(json_path,'w') as fp:
            json.dump(data, fp, indent=4)

    

if __name__ == '__main__':
    save_mfcc(dataset_path=DATASET_PATH,json_path=JSON_PATH)