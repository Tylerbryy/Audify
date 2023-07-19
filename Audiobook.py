import os
import torch
from PyPDF2 import PdfReader
from pydub import AudioSegment
from urllib.request import urlretrieve
import streamlit as st

# Title and Instructions
st.markdown("# 📚 PDF to Audiobook Converter")
st.markdown("Upload a PDF, select a speaker, then click Convert to Audiobook. (Please note the bigger the book the longer the process will take)")

# Settings sidebar
st.sidebar.header("Settings")
speaker = st.sidebar.selectbox('Select a Speaker', [f'en_{i}' for i in range(1, 118)])

# File uploader
pdf_file_path = st.file_uploader('Upload PDF', type='pdf')

def generate_audio_chunk(chunk, speaker, sample_rate, model):
    try:
        audio_paths = model.save_wav(text=chunk, speaker=speaker, sample_rate=sample_rate)
        chunk_audio = AudioSegment.from_wav(audio_paths)
        return chunk_audio
    except Exception as e:
        
        return None

def text_to_audio_book(pdf_file, speaker):
    device = torch.device('cpu')
    torch.set_num_threads(torch.get_num_threads())
    local_file = 'v3_en.pt'

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    sample_rate = 48000

    pdf_reader = PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    progress_bar = st.progress(0)

    full_audio = AudioSegment.empty()

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

            full_audio += chunk_audio

        st.write(f"✔️ Audio for page {page_num} saved.")
        progress_bar.progress((page_num + 1) / num_pages)

    full_audio.export("full_audio.wav", format="wav")
    progress_bar.empty()

# Check if model is installed, if not download it.
if not os.path.isfile('v3_en.pt'):
    urlretrieve('https://models.silero.ai/models/tts/en/', 'v3_en.pt')  # replace 'model_download_url' with actual URL of the model

if st.button('Convert to Audiobook'):
    if pdf_file_path is not None:
        text_to_audio_book(pdf_file_path, speaker)
        st.success('Successfully converted PDF to Audiobook. 🎉')
        st.audio('full_audio.wav')
    else:
        st.error('Please upload a PDF file before proceeding.')