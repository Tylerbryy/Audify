# 📚 PDF to Audiobook Converter

This Python script allows you to convert a PDF into an audiobook. You can upload a PDF, select a speaker, and then convert the PDF to an audiobook. Please note that the larger the book, the longer the process will take.

![alt text for screen readers](https://media.giphy.com/media/WoWm8YzFQJg5i/giphy.gif "Text to show on mouseover").


## Requirements

This script requires the following Python libraries:

- os
- torch
- wave
- numpy
- PyPDF2
- pydub
- urllib.request
- streamlit
- logging

You can install these libraries using pip:

```bash
pip install -r requirements.txt
```
##  Usage
Run the script using Streamlit:
```bash
streamlit run Audiobook.py
```
- In the Streamlit interface, select a speaker from the dropdown menu.
- Upload your PDF file.
- Click the 'Convert to Audiobook' button.
- Once the conversion is complete, you can download your audiobook.

## Note
If the model is not installed, the script will automatically download it from 'https://models.silero.ai/models/tts/en/' and save it as 'v3_en.pt'