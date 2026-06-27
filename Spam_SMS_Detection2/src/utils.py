import re

def clean_text(text):
    """
    Clean SMS text for ML model:
    - lower case conversion
    - remove special characters
    - remove extra spaces
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()  
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    text = text.strip()  

    return text