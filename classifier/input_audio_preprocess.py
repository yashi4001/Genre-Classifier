import librosa
import math
import json
import os

FILE="your-summer-day-5448.wav"
JSON_PATH="audio.json"
SAMPLE_RATE=22050
DURATION=30 #measuring duration of song in seconds as per dataset
SAMPLES_PER_TRACK=SAMPLE_RATE*DURATION

def save_mfcc(file,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
    #dictionary to store data
    data={
        "mfcc":[]
    }
    num_samples_per_segment=int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment=math.ceil(num_samples_per_segment/hop_length)

    signal,sr=librosa.load(file,sr=SAMPLE_RATE)

    #process segments extracting mfcc and storing mfcc value
    for s in range(num_segments):
        start_sample=num_samples_per_segment*s
        finish_sample=start_sample+num_samples_per_segment

        mfcc=librosa.feature.mfcc(signal[start_sample:finish_sample],sr=sr,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)

        mfcc=mfcc.T

        #store mfcc for segment if it has the expected length
        if len(mfcc)==expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
    
    #storing data in json file                    
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)
                


# if __name__=="__main__":
#     save_mfcc(file,JSON_PATH,num_segments=10)

            
