import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

#%% Function to load and process the uploaded audio file
def process_audio(original_audio, orig_sr, sampling_rate, quantization_levels):
    
    resampled_audio = librosa.resample(original_audio, orig_sr=orig_sr, target_sr=sampling_rate)

    quantized_audio = np.round(resampled_audio * (2 ** quantization_levels)) / (2 ** quantization_levels)
    
    return quantized_audio

# Streamlit app
st.divider()
st.title("Audio Sampling and Quantization Demo \t")
st.divider()

# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

# Sampling rate and quantization levels
sampling_rate = st.sidebar.slider("Select sampling rate (Hz)", min_value=500, max_value=48000, value=44100)
quantization_levels = st.sidebar.slider("Select quantization (bits)", min_value=1, max_value=16, value=8)

# Main Streamlit app
def main():
    
    if uploaded_file:

        if sampling_rate or quantization_levels:
            
            original_audio, orig_sr  = librosa.load(uploaded_file)
            
            # Process the uploaded audio
            quantized_audio = process_audio(original_audio, orig_sr, sampling_rate, quantization_levels)
            np.nan_to_num(quantized_audio,nan= 0, posinf = 999999)
            
            #%% Compute the FFT
            original_audio_FFT  = np.abs(np.fft.rfft(original_audio)) * 2/len(original_audio)
            quantized_audio_FFT = np.abs(np.fft.rfft(quantized_audio)) * 2/len(quantized_audio)

            # Compute the frequency axis
            frequencies_org = np.linspace(0, orig_sr/2, len(original_audio_FFT))
            frequencies_quant = np.linspace(0, sampling_rate/2, len(quantized_audio_FFT))
            
            #%% Display original audio
            st.subheader(f"Original Audio Signal (Rate: {orig_sr})")
            st.audio(original_audio, format="wav",sample_rate=orig_sr)
        
            # Display quantized audio
            st.subheader(f"Sampled (Rate: {sampling_rate}) and Quantized Audio Signal (Levels: {quantization_levels})")
            st.audio(quantized_audio, format="wav",sample_rate=sampling_rate)
        
            #%% Plot time waveforms
            st.subheader("Waveforms")            
            t = np.linspace(0, len(original_audio) / orig_sr, len(original_audio))            
            
            fig, axs = plt.subplots(1,2)
            # plt.figure(figsize=(8, 4))
            axs[0].plot(t, original_audio)     
            axs[0].set_title("Original")
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Amplitude")            
            axs[0].grid()           
            
            t_new = np.linspace(0, len(quantized_audio) / sampling_rate, len(quantized_audio))
            axs[1].plot(t_new, quantized_audio)
            axs[1].set_title("Sampled and Quantized")
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
            axs[0].grid() 
            
            axs[1].plot(frequencies_quant, quantized_audio_FFT, linewidth=2, color='blue')
            axs[1].set_title('Sampled and Quantized')
            axs[1].set_xlabel('Frequency (Hz)')            
            axs[1].set_ylabel("Magnitude") 
            axs[1].grid()
            st.pyplot(fig)

if __name__ == "__main__":
    main()


