from google import genai
from google.genai import types

client = genai.Client(api_key='AIzaSyBrFThAoa_2we3ywzfs2YCCvzPRUyannCo')

response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)