import os
import torch
import wave
import numpy as np
from PyPDF2 import PdfReader
from pydub import AudioSegment
from urllib.request import urlretrieve
import streamlit as st
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Title and Instructions
st.markdown("# 📚 PDF to Audiobook Converter")
st.markdown("Upload a PDF, select a speaker, then click Convert to Audiobook. (Please note the bigger the book the longer the process will take)")

# Select a speaker
speaker = st.selectbox('Select a Speaker', [f'en_{i}' for i in range(1, 118)])

# File uploader
uploaded_file = st.file_uploader('Upload PDF', type='pdf')

if uploaded_file is not None:
    book_name = uploaded_file.name.split('.')[0]  # Assuming no '.' in the book name
else:
    book_name = None

def generate_audio_chunk(chunk, speaker, sample_rate, model):
    try:
        audio_paths = model.save_wav(text=chunk, speaker=speaker, sample_rate=sample_rate)
        chunk_audio = AudioSegment.from_wav(audio_paths)
        return chunk_audio
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        # If the chunk is too long, halve it and try again
        if "too long" in str(e) and len(chunk) > 1:
            return generate_audio_chunk(chunk[:len(chunk) // 2], speaker, sample_rate, model)
        return None


def text_to_audio_book(uploaded_file, speaker):
    device = torch.device('cpu')
    torch.set_num_threads(torch.get_num_threads())
    local_file = 'v3_en.pt'

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    sample_rate = 48000

    pdf_reader = PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    progress_bar = st.progress(0)

    if book_name is not None:
        audiobook_file = f"{book_name}_audiobook.wav"
    else:
        audiobook_file = "audiobook.wav"

    wavef = wave.open(audiobook_file,'w')
    wavef.setnchannels(1) # mono
    wavef.setsampwidth(2) 
    wavef.setframerate(sample_rate)

    for page_num in range(num_pages):
        text = pdf_reader.pages[page_num].extract_text().replace('\n', ' ')

        if text == '':
            st.write(f"⚠️ No text found on page {page_num}, skipping this page.")
            continue

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            while True:
                chunk_audio = generate_audio_chunk(chunk, speaker, sample_rate, model)
                if chunk_audio is not None:
                    break
                # Retry with a smaller chunk size
                chunk = chunk[:len(chunk) // 2]

            wavef.writeframesraw(np.array(chunk_audio.get_array_of_samples()))

        st.write(f"✔️ Audio for page {page_num} saved.")
        progress_bar.progress((page_num + 1) / num_pages)

    wavef.close()
    progress_bar.empty()

    # Download button
    with open(audiobook_file, "rb") as file:
        btn = st.download_button(
            label="Download Audiobook",
            data=file,
            file_name=audiobook_file,
            mime="audio/wav"
        )

# Check if model is installed, if not download it.
if not os.path.isfile('v3_en.pt'):
    urlretrieve('https://models.silero.ai/models/tts/en/', 'v3_en.pt')  

if st.button('Convert to Audiobook'):
    if uploaded_file is not None:
        text_to_audio_book(uploaded_file, speaker)
        st.success(f'Successfully converted {book_name} to Audiobook. 🎉')
    else:
        st.error('Please upload a PDF file before proceeding.')
