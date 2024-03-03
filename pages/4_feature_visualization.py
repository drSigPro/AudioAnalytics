import librosa
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Streamlit app
st.divider()
st.title("Demo for visualizing features\t")
st.divider()


# Upload an audio file
uploaded_file1 = st.file_uploader("Upload first audio file (WAV format)", type=["wav"])
uploaded_file2 = st.file_uploader("Upload second audio file (WAV format)", type=["wav"])

  
# Main Streamlit app
def main():
    
    if uploaded_file1 and uploaded_file2:

        if st.button('Visualize'):
            
            # Load audio data
            y1, sr1 = librosa.load(uploaded_file1)
            y2, sr2 = librosa.load(uploaded_file2)

            len_y = min(len(y1),len(y2))
            y1_new = librosa.resample(y1, orig_sr=sr1, target_sr=sr1*len(y1)/len_y)
            y2_new = librosa.resample(y2, orig_sr=sr2, target_sr=sr2*len_y/len(y2))
            
            y1 = y1_new[0:len_y-1]
            y2 = y2_new[0:len_y-1]
            
            stft1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1,hop_length=128, n_fft=1024)))
            stft2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2,hop_length=128, n_fft=1024)))
            
            mtft1 = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(y=y1, sr=sr1, hop_length=128, n_fft=1024)))
            mtft2 = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(y=y2, sr=sr2, hop_length=128, n_fft=1024)))
            
            mfcc1 = librosa.feature.mfcc(y=y1, n_mfcc=100, sr=sr1,dct_type=2)
            mfcc2 = librosa.feature.mfcc(y=y2, n_mfcc=100, sr=sr2,dct_type=2)

             st.subheader("Comparison")                    
            fig, ax = plt.subplots(4,1)
            ax[0].scatter(y1, y2, c='b', label='Raw', marker = 'o')
            ax[0].set_title('Raw Signals')
            ax[1].scatter(stft1, stft2, c='r', label='STFTs', marker = 'o')
            ax[1].set_title('STFTs')
            ax[2].scatter(mtft1,mtft2, c='g', label='STMTs', marker = 'o')
            ax[2].set_title('Mel-Spectrograms')
            ax[3].scatter(mfcc1, mfcc2, c='k', label='STFTs', marker = 'o')
            ax[3].set_title('MFCCs')            
            st.pyplot(fig)
            
if __name__ == "__main__":
    main()


