# Accent classification for improving accents in foreign languages using fast.ai and resnet18

1. Introduction
2. Data Processing
3. Training the model
4. Results
5. Conclusion and thoughts

{:introduction}
I am learning french at the moment and one of the things I would like to improve is my pronunciation. The problem is that it's hard to get feedback when you are practicing by yourself. If there was a computer program where you can input your self speaking and it would tell you whether the accent was french or not, then that would be good feedback. So the purpose of this model is to try take in audio recordings of me speaking french and to classify whether the accent was french or english.

{:Data Collection}
## Data Collection

I searched online and Mozilla have a project called Mozilla Common Voice which have free to use datasets of people speaking in various languages. Under the dataset tab I searched for english and french datasets and downloaded around 1Gb dataset for english and a 2Gb data set for french.

~[]{/images/blog_2024_04_23_accent_recognition/mozilla_common_voice_dataset.png}

I extracted the dataset into a folder on my computer. 

{;Data Processing}
## Data Processing

resnet18 is an image processing model not an audio processing model so I needed to convert my audio files to image files. What you get when you did this is an image called a spectrogram. Python has some simple libraries to help me do this however to do so I needed to use .wav audio files however the data set downloaded from mozilla common voice contained only .mp3 files.

Using ffmpeg I converted the english and french .mp3 files to .wav files. After that I wrote a small python script that would take in all the .wav files and convert them to spectroms (code is shown below).

    import matplotlib.pyplot as plt
    from scipy import signal
    from scipy.io import wavfile
    import numpy as np
    import os

    def create_spectrogram(filepath):
        # Load stereo audio file
        sample_rate, audio = wavfile.read(filepath)

        # My .wav files were all mono channel. If you had wav files
        # that are dual channel you will need to convert them to
        # a single channel by uncommenting the code below
        #mono = [0]
        #for i in range(len(audio)):
        #    mono.append(audio[i][0])

        # Convert list to NumPy array
        np_array = np.array(audio)
        
        # Compute spectrogram
        freq, times, Sxx = signal.spectrogram(np_array, sample_rate)


        # Add a small constant to avoid log(0) or log(negative_number)
        epsilon = 1e-10
        Sxx[Sxx <= 0] = epsilon

        # Plot spectrogram without labels, titles, and color bar
        plt.figure(figsize=(8, 4))
        plt.imshow(10 * np.log10(Sxx), aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')  # Turn off axis
        plt.tight_layout()

        # Save the image
        filename = os.path.splitext(os.path.basename(filepath))[0]
        plt.savefig("harry_potter_spectrographs/" + filename + "_spectrogram.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    path = "Data/french_wavs/"

    dir_list = os.listdir(path)
    for file in dir_list:
        create_spectrogram(path + file)

An example spectrogram I got from one of the .wav files is shown below. Note if yours looks like a bunch of horizontal lines instead of vertical, your audio file is probably dual channel and you'll need to convert it to single channel before creating the spectrogram.

![](/images/spectrogram_example.png "Example spectrogram of audio recording")

Once your here you should have two folders one containing all the spectrograms of the french audio and one with the english audio.

{:Training the model}
## Training the model

I made all the spectrograms on my laptop but this doesn't have a GPU so I used one of the lab computers at my universtity that has a GPU to help speed up the training time. Below are the screenshots of my jupyter notebook that contains all the code I used to traing the model

![](/images/notebook_img1.png "Example spectrogram of audio recording")
![](/images/notebook_img2.png "Example spectrogram of audio recording")
![](/images/notebook_img3.png "Example spectrogram of audio recording")
![](/images/notebook_img4.png "Example spectrogram of audio recording")

Now I am creating the model using fast.ai library. This is my first time doing anything deep learning related so I don't anything about what's going on under the hood but from what I heard, the fast.ai function picks all the settings for you to more or less get a pretty good result for what you need so I just followed their examples online.

I am also setting the batch size to 256 here to speed up the training time. The last thing I am doing is showing some samples in the batch to verify that it has the right images
![](/images/notebook_img5.png "Example spectrogram of audio recording")

Next thing to do is train the model. Again this is something more or less handled by fast.ai, you just call the function and it does it for you. I chose to 25 epochs. If you look at epoch 21 you'll see it has the lowest valid loss which is what we are looking for (or so I was told). For the most accurate model I think you would go back and retrain the model using 21 instead of 25 but I just kept it with 25.

![](/images/notebook_img6.png "Example spectrogram of audio recording")

The next thing to do is print out how well the model did. Looking at the images of which ones it got confused on doesn't really help us with spectrgrams because I can't tell from looking at them which ones are french audio and which ones are english audio so that's not all the useful. The confusion matrix was interesting though as it showed that when validating the model it got most of it right. I have no idea what patterns it's picking up in the images but it's certainly picking up on something!

![](/images/notebook_img7.png "Example spectrogram of audio recording")
![](/images/notebook_img8.png "Example spectrogram of audio recording")

{:results}
## Results

The last thing to do is test the model with some real data. I have the audio book of Harry Potter in french on audible so I recorded on my laptop the narrator speaking the first two sentences of the first chapter. I then converted the audio file to a spectrogram and labelled it as 'good'. I then recorded myself saying the first two sentences in my best possible french accent and then did it again in an english accent. I made two more spectrograms of those recordings labelled 'ok' and 'bad'. I then ran them through the predictor to see what the results were.

As you can below it labelled them all correctly which was promosing although I didn notice it wasn't very confident when labelling the 'good' and 'ok' spectrograms as french. This isn't a problem on the 'ok' spectrogram because my best french accent isn't very french but on the 'good' one I thought it was a bit odd it had such a weak confidence for it even though if you listened to the recoding it is 100% french without a doubt.

![](/images/notebook_img9.png "Example spectrogram of audio recording")

Given the first test I did showed promise I went and recorded some more samples of good, ok and bad pronunication of the french language, made the spectrograms and then got the model to classify the accent. Below are the results.

You can see that with more test samples that the model started to make a some errors but on the whole it does seem to be getting the general gist of it.

![](/images/notebook_img10.png "Example spectrogram of audio recording")

{:Conclusion and thoughts}
## Conclusion and thoughts

Below are some thoughts on how I could improve the model/maybe some things I need to explore

- The data that the model was trained on was french people speaking french and english people speaking **english**. This ideal case would be getting data on foreigners speaking french with poor french accents however that wasn't available so I had to make do with english speaking english accents.
- I'm not quite sure why the confidence for predicting the native french speaker as having a french accent was so low. That is something I will need to explore and don't have any answers at the moment
- I didn't really examine the data set that much because **1)** There was so much data and **2)** I didn't really have a lot of time but I could try comb through and cull out any audio that is kind of muffled or has a poor accent for that language. The mozilla commons voice dataset is there to recognise the language being spoken so they have many different types of accents all saying the language to get a good spread where as I need more narrow range of data where there is only people speaking with accents that I want labelled as one type and people speaking with accents that I don't want labelled as the other type. I could try that but there were over 70,000 recordings in the each data set so without an automated way to do that, it would take quite a long time. Ironically, I could do it automatically with the thing I am trying to develop if I had working very well but I don't so i'd have to do it manually. What I could potentially do is exmine what spectrograms it got confused on the most and look at the audio recording they corresponded to to try identify what about the audio recording made it get confused.
