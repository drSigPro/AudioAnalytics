import librosa
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import numpy as np

# Streamlit app
st.divider()
st.title("Demo for visualizing features using t-SNE \t")
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
            
            stft1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1,hop_length=128, n_fft=1024)))
            stft2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2,hop_length=128, n_fft=1024)))
            stft_concatenated = np.concatenate((stft1, stft2), axis=1)

            
            mtft1 = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(y=y1, sr=sr1, hop_length=128, n_fft=1024)))
            mtft2 = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(y=y2, sr=sr2, hop_length=128, n_fft=1024)))
            mtft_concatenated = np.concatenate((mtft1, mtft2), axis=1)

            
            mfcc1 = librosa.feature.mfcc(y=y1, n_mfcc=100, sr=sr1,dct_type=2)
            mfcc2 = librosa.feature.mfcc(y=y2, n_mfcc=100, sr=sr2,dct_type=2)
            mfcc_concatenated = np.concatenate((mfcc1, mfcc2), axis=1)
            
            #%% Display Raw Signal Relation
            st.subheader("Raw Signal Comparison")
            len_y = min(len(y1),len(y2))            
            fig, ax = plt.subplots()
            ax.plot(y1[0:len_y-1],y2[0:len_y-1],'bo')
            ax.set_title("Raw Signal")
            st.pyplot(fig)
            
            #%% Apply t-SNE
            tsne = TSNE(n_components=2)
            stft_tsne = tsne.fit_transform(stft_concatenated.T)
            mtft_tsne = tsne.fit_transform(mtft_concatenated.T)
            mfcc_tsne = tsne.fit_transform(mfcc_concatenated.T)

            #%% Apply UMAP
            #reducer  = umap.UMAP(n_components=2)
            #stft_umap = reducer.fit_transform(stft_concatenated.T)
            #mtft_umap = reducer.fit_transform(mtft_concatenated.T)
            #mfcc_umap = reducer.fit_transform(mfcc_concatenated.T)
            
            
            st.subheader("tSNE Comparison")                    
            fig, ax = plt.subplots(3,1)
            ax[0].scatter(stft_tsne[:, 0], stft_tsne[:, 1], c='b', label='STFTs')
            ax[0].set_title('STFTs')
            ax[1].scatter(mtft_tsne[:, 0], mtft_tsne[:, 1], c='b', label='STMTs')
            ax[1].set_title('Mel-Spectrograms')
            ax[2].scatter(mfcc_tsne[:, 0], mfcc_tsne[:, 1], c='b', label='STFTs')
            ax[2].set_title('MFCCs')
            st.pyplot(fig)
            
           # st.subheader("UMAP Comparison")                    
           # fig, ax = plt.subplots(3,1)
           # ax[0].scatter(stft_umap[:, 0], stft_umap[:, 1], c='b', label='STFTs')
           # ax[0].set_title('STFTs')
           # ax[1].scatter(mtft_tsne[:, 0], mtft_umap[:, 1], c='b', label='STMTs')
           # ax[1].set_title('Mel-Spectrograms')
           # ax[2].scatter(mfcc_umap[:, 0], mfcc_umap[:, 1], c='b', label='STFTs')
           # ax[2].set_title('MFCCs')
           # st.pyplot(fig)  
            
if __name__ == "__main__":
    main()


