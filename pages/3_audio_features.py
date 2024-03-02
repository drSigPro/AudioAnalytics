import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

def amplitude_envelope(signal, window, overlap):
    amplitude_envelope = []
    
    for i in range(0, len(signal), overlap): 
        envelope_temp = max(signal[i:i+window]) 
        amplitude_envelope.append(envelope_temp)
    
    return np.array(amplitude_envelope) 

# Streamlit app
st.divider()
st.title("Demo for Extracting Audio Features \t")
st.divider()


# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])




option = st.sidebar.selectbox('Filter Type',('time', 'frequency', 'time-frequency'))

if (option=='time'):
    windowlength = st.sidebar.slider("Select Window Length", min_value=100, max_value=10000, value=1024)
    hopsize = st.sidebar.slider("Select Hop Size", min_value=100, max_value=10000, value=512)
    
if (option=='frequency'):
    windowlength = st.sidebar.slider("Select Window Length", min_value=100, max_value=10000, value=1024)
    hopsize = st.sidebar.slider("Select Hop Size", min_value=100, max_value=10000, value=512)
    fftsize = st.sidebar.slider("Select FFT Size", min_value=128, max_value=2048, value=1024)
    
if (option=='time-frequency'):
    windowlength = st.sidebar.slider("Select Window Length", min_value=100, max_value=10000, value=1024)
    hopsize = st.sidebar.slider("Select Hop Size", min_value=100, max_value=10000, value=512)
    fftsize = st.sidebar.slider("Select FFT Size", min_value=128, max_value=2048, value=1024)
    
# Main Streamlit app
def main():
    
    if uploaded_file:

        if option:
            
            original_audio, orig_sr  = librosa.load(uploaded_file)
            t = np.linspace(0, len(original_audio) / orig_sr, len(original_audio))           
            
           
            if (option=='time'):
                envelope = amplitude_envelope(original_audio, windowlength, hopsize)
                t_envelope = np.arange(len(envelope)) * (len(original_audio) / len(envelope)) / orig_sr
                zcr = librosa.feature.zero_crossing_rate(original_audio,frame_length=windowlength,hop_length=hopsize)[0]
                t_zcr = np.arange(len(zcr)) * (len(original_audio) / len(zcr)) / orig_sr
                rms = librosa.feature.rms(y=original_audio,frame_length=windowlength,hop_length=hopsize)[0]
                t_rms = np.arange(len(rms)) * (len(original_audio) / len(rms)) / orig_sr

                #%% Plot time waveforms
                fig, axs = plt.subplots(3, 1, figsize=(15, 10))
                #st.subheader("Envelope") 
                #fig, ax = plt.subplots(1,1)
                axs[0].plot(t, original_audio)   
                axs[0].plot(t_envelope, envelope)                 
                axs[0].set_xlabel("Time (s)")
                axs[0].set_ylabel("Amplitude") 
                axs[0].set_title("Envelope")
                axs[0].grid()
                #st.pyplot(fig)
                
                #st.subheader("Zero Crossing Rate") 
                #fig, ax = plt.subplots(1,1)
                axs[1].plot(t, original_audio)   
                axs[1].plot(t_zcr, zcr)                 
                axs[1].set_xlabel("Time (s)")
                axs[1].set_ylabel("Amplitude")
                axs[1].set_title("Zero Crossing Rate")
                axs[1].grid()
                #st.pyplot(fig)
                
                #st.subheader("Root Mean Square") 
                #fig, ax = plt.subplots(1,1)
                axs[2].plot(t, original_audio)   
                axs[2].plot(t_rms, rms)                 
                axs[2].set_xlabel("Time (s)")
                axs[2].set_ylabel("Amplitude")  
                axs[2].set_title("Root Mean Square")
                axs[2].grid()
                st.pyplot(fig)
            
            if (option=='frequency'):              
                
                #%% Spectral Centroid
                cent = librosa.feature.spectral_centroid(y=original_audio, sr=orig_sr,n_fft=fftsize, hop_length=hopsize, win_length=windowlength)
                t_cent = np.arange(len(cent[0])) * (len(original_audio) / len(cent[0])) / orig_sr
               
                
                st.subheader("Spectral Centroid") 
                fig, ax = plt.subplots(1,1)
                ax.plot(t, original_audio)  
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")  
                ax.grid()
                ax2 = ax.twinx()
                ax2.plot(t_cent, cent.T,'r+')                 
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Frequency *Hz)")  
                # ax2.grid()
                st.pyplot(fig)
                
            if (option=='time-frequency'):
                
                              
                #%% Spectrogram
                num_windows = len(original_audio) // hopsize
                # Create an array to store FFTs
                all_ffts = np.zeros((num_windows, windowlength // 2 + 1), dtype=complex)

                # Apply a window to the time series
                window = np.hanning(windowlength)
                
                for i in range(num_windows-1):
                    start = i * hopsize
                    end = start + windowlength
                    windowed_signal = original_audio[start:end] * window
                    windowed_fft = np.fft.fft(windowed_signal)
                    all_ffts[i, :] = windowed_fft[:windowlength // 2 + 1]
                
                # Convert magnitudes to decibels
                all_ffts_db = librosa.amplitude_to_db(np.abs(all_ffts), ref=np.max)
                
                #%% Spectral Contrast                
                S = np.abs(librosa.stft(y=original_audio,n_fft=fftsize, hop_length=hopsize, win_length=windowlength))
                contrast = librosa.feature.spectral_contrast(S=S, sr=orig_sr)
                
                #%% Mel Spectrogram
                mel_signal = librosa.feature.melspectrogram(y=original_audio, sr=orig_sr, hop_length=hopsize, n_fft=fftsize)
                spectrogram = np.abs(mel_signal)
                power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
                
                #%% MFCC
                mfccs = librosa.feature.mfcc(y=original_audio, n_mfcc=100, sr=orig_sr,dct_type=2)
                
                # Display the results
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                ax = axs[0,0]
                librosa.display.specshow(all_ffts_db.T, sr=orig_sr, hop_length=hopsize, x_axis='time', y_axis='linear', ax = axs[0,0])
                ax.set_title('Spectrogram')                
                ax.set_xlabel('Time')
                ax.set_ylabel('Frequency (Hz)')

                ax = axs[0, 1]
                librosa.display.specshow(contrast, x_axis="time",sr=orig_sr,ax = axs[0, 1])                              
                ax.set(title='log Power spectrogram')

                ax = axs[1,0]
                librosa.display.specshow(power_to_db, sr=orig_sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hopsize,ax = axs[1,0])
                ax.set_title('Mel-Spectrogram (dB)')
                ax.set_xlabel('Time')
                ax.set_ylabel('Log Frequency')
                
                ax = axs[1,1]
                librosa.display.specshow(mfccs, x_axis="time",sr=orig_sr,ax = axs[1,1])
                ax.set_title('MFCC')
                ax.set_xlabel('Time')
                ax.set_ylabel('MFCC')
            
                st.pyplot(fig)
                
if __name__ == "__main__":
    main()


