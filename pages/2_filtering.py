import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt,freqz

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    w, h = freqz(b, a, worN=8000)
    # Plot the magnitude response
    fig, ax = plt.subplots(1,1)
    ax.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), 'b')
    ax.set_title('Bandpass Filter Frequency Response')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid()
    st.pyplot(fig)
    
    return b, a

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq   
    b, a = butter(order, low, btype='low')
    
    w, h = freqz(b, a, worN=8000)
    # Plot the magnitude response
    fig, ax = plt.subplots(1,1)
    ax.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), 'b')
    ax.set_title('Lowpass Filter Frequency Response')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid()
    st.pyplot(fig)
    
    return b, a

def butter_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq   
    b, a = butter(order, high, btype='high')
    
    w, h = freqz(b, a, worN=8000)
    # Plot the magnitude response
    fig, ax = plt.subplots(1,1)
    ax.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), 'b')
    ax.set_title('Highpass Filter Frequency Response')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid()
    st.pyplot(fig)
    
    return b, a

# Streamlit app
st.divider()
st.title("Pre-processing (Filtering) Demo \t")
st.divider()


# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

option = st.sidebar.selectbox('Filter Type',('high', 'low', 'band'))

if (option=='low'):
    lowcut = st.sidebar.slider("Select cut-off frequency (Hz)", min_value=10, max_value=10000, value=5000)
    
if (option=='high'):
    highcut = st.sidebar.slider("Select cut-off frequency (Hz)", min_value=10, max_value=10000, value=5000)
    
if (option=='band'):
    lowcut = st.sidebar.slider("Select lower cut-off frequency (Hz)", min_value=10, max_value=10000, value=5000)
    highcut = st.sidebar.slider("Select higher cut-off frequency (Hz)", min_value=10, max_value=10000, value=5000)
    
# Main Streamlit app
def main():
    
    if uploaded_file:

        if option:
            
            original_audio, orig_sr  = librosa.load(uploaded_file)
            
            # filter the uploaded audio
            if (option=='low'):
                b,a = butter_lowpass(lowcut, orig_sr, order=5)
                
                
            if (option=='high'):
                b,a = butter_highpass(highcut, orig_sr, order=5)
                
            if (option=='band'):                
                b,a = butter_bandpass(lowcut,highcut, orig_sr, order=5)
                
                
                
            filtered_audio = filtfilt(b, a, original_audio)
            
            #%% Compute the FFT
            original_audio_FFT  = np.abs(np.fft.rfft(original_audio)) * 2/len(original_audio)
            filtered_audio_FFT = np.abs(np.fft.rfft(filtered_audio)) * 2/len(filtered_audio)

            # Compute the frequency axis
            frequencies_org = np.linspace(0, orig_sr/2, len(original_audio_FFT))
            
            #%% Display original audio
            st.subheader("Original Audio Signal")
            st.audio(original_audio, format="wav",sample_rate=orig_sr)
        
            # Display quantized audio
            st.subheader("Filtered Audio Signal")
            st.audio(filtered_audio, format="wav",sample_rate=orig_sr)
        
            #%% Plot time waveforms
            st.subheader("Waveforms")            
            t = np.linspace(0, len(original_audio) / orig_sr, len(original_audio))            
            
            fig, axs = plt.subplots(1,2)
            axs[0].plot(t, original_audio)     
            axs[0].set_title("Original")
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Amplitude")  
            axs[0].grid()
                       
            
            axs[1].plot(t, filtered_audio)
            axs[1].set_title("Filtered")
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Amplitude")    
            axs[1].grid()
            st.pyplot(fig)

            #%% Plot the FFTs
            st.subheader("Spectrum")  
            fig, axs = plt.subplots(1,2)
            axs[0].plot(frequencies_org, original_audio_FFT, linewidth=2, color='blue')
            axs[0].set_title('Original')
            axs[0].set_xlabel('Frequency (Hz)')            
            axs[0].set_ylabel("Magnitude") 
            axs[0].grid(visible=True, which='major', axis='both')
            
            axs[1].plot(frequencies_org, filtered_audio_FFT, linewidth=2, color='blue')
            axs[1].set_title('Sampled and Quantized')
            axs[1].set_xlabel('Frequency (Hz)')            
            axs[1].set_ylabel("Magnitude") 
            axs[1].grid(visible=True, which='major', axis='both')
            st.pyplot(fig)

if __name__ == "__main__":
    main()


