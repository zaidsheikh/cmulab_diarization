# from demo_utils import *
from pathlib import Path
import json
import itertools
import numpy as np
import librosa
import argparse
from .resemblyzer import preprocess_wav, VoiceEncoder, sampling_rate
#from diarization import audio_length

# from resemblyzer import preprocess_wav, VoiceEncoder
# from demo_utils import *
# from pathlib import Path
# import json
# import itertools
# import numpy as np
# import librosa


# method to return the values of same items in a list
def find_num_of_split(list, i):
    repeating_list=[]
    if list.count(i)>0:
        # repeating_list=[i, list.count(x)]
        repeating_list=[i, list.count(i)]
        return repeating_list


# DEMO 02: we'll show how this similarity measure can be used to perform speaker diarization
# (telling who is speaking when in a recording).

## Get reference audios
# Load the interview audio from disk
# Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I

def run_diarization(input_file_path, segments, speaker_names, threshhold=0.45):

    wav_fpath = Path(input_file_path)
    wav = preprocess_wav(wav_fpath)
    audio_length= librosa.get_duration(filename=wav_fpath)

    # Cut some segments from single speakers as reference audio
    speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]


    ## Compare speaker embeds to the continuous embedding of the interview
    # Derive a continuous embedding of the interview. We put a rate of 16, meaning that an
    # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker
    # diarization, but it is not so useful for when you only need a summary embedding of the
    # entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the
    # demonstration.
    # We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs
    # won't have enough. There's a speed drawback, but it remains reasonable.


    encoder = VoiceEncoder("cpu")
    print("Running the continuous embedding on cpu, this might take a while...")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)


    # Get the continuous similarity for every speaker. It amounts to a dot product between the
    # embedding of the speaker and the continuous embedding of the interview
    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in
                       zip(speaker_names, speaker_embeds)}



    # creating a list with ['UNK', 0] with the size of length equal to similarity_dict
    val = 'UNK'
    size = len(list(similarity_dict.values())[0])
    similarity_list= list(itertools.repeat(0, size))
    speaker_on_splits = list(itertools.repeat(val, size))


    # the list will have the maximum values for corresponding column of values for each keys
    # speaker_on_splits will have the values that which speaker spoke on which wav split
    for key, values in similarity_dict.items():
        i=0
        for value in values:
            if similarity_list[i]<value and value>threshhold:
                similarity_list[i] = value
                speaker_on_splits[i] = key
            i=i+1

    # calculating the length of each split by diving the audio length with number of splits.
    split_length = audio_length / len(speaker_on_splits)


    # number of splits a speaker spoke continuously
    continuous_speaker_list=speaker_on_splits

    temp_list2=[]

    for x in speaker_on_splits:
        temp_list=[]
        count=0
        for y in continuous_speaker_list:
            if x==y:
                temp_list.append(x)
                count=count+1
            else:
                break
        for i in range(count):
            continuous_speaker_list = np.delete(continuous_speaker_list, 0)
        temp_list2.append(find_num_of_split(temp_list, x))

    speaker_list_with_splits_number = [i for i in temp_list2 if i]

    # creating the list with timestamps
    speaker_list_with_timestamp = []
    time=0
    for x in speaker_list_with_splits_number:
        speaker_list_with_timestamp.append( [x[0], float(time), float((time+x[1]*split_length))])
        time= x[1]*split_length+time

    # # output in the json file
    # with open ("speaker_list_with_timestamp.json", "w") as wf:
        # json.dump(speaker_list_with_timestamp, wf)

    # print(json.dumps(speaker_list_with_timestamp, indent=4))
    return speaker_list_with_timestamp

    # speaker_list_in_wavs=[]

    # time=0
    # for x in speaker_on_splits:
        # speaker_list_in_wavs.append([x, float(time*1000), float((time+split_length)*1000)])
        # time= time+split_length

    # with open ("speaker_list_in_wavs_hyp.json", "w") as wf:
        # json.dump(speaker_list_in_wavs, wf)


if __name__ == "__main__":
    audio_file = "audio_data/X2zqiX6yL3I.mp3"
    segments = [[0, 5.5], [6.5, 12], [17, 25]]
    names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    run_diarization(audio_file, segments, names)
