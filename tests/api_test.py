import os
import requests


def test_event_endpoint():
    """
    Tests the /v1/event endpoint by sending a multipart request.
    """
    # --- Configuration ---
    BENTO_URL = "http://localhost:3000/predict"
    API_KEY = "test_api_key"  # Must match the key in your service
    MODEL_TO_USE = "med-general-1" # Must match a runner defined in your service

    print("--- Preparing Test Request ---")
    
    # 1. Prepare the request headers for authentication
    headers = {
        "X-Api-Key": API_KEY
    }
    print(f"Using API Key: {API_KEY}")

    LOCAL_AUDIO_FILE_PATH = os.path.join(os.getcwd(),  "tests","2024-12-19_13.44.16.919__v0.wav")

    # 2. Prepare the multipart form data
    with open(LOCAL_AUDIO_FILE_PATH, "rb") as audio_file:
            # The 'files' dictionary takes the filename, the file object itself,
            # and the content type.
            files = {
                'audio_file': (os.path.basename(LOCAL_AUDIO_FILE_PATH), audio_file, 'audio/wav')
            }

            form_data = {
                'model': MODEL_TO_USE
            }
            print(f"Using model: '{MODEL_TO_USE}'")

            # --- Send the Request ---
            print(f"\nSending POST request to {BENTO_URL}...")
            try:
                response = requests.post(BENTO_URL, headers=headers, files=files, data=form_data)

                # --- Process the Response ---
                print(f"\n--- Server Response ---")
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    print("Request successful!")
                    print("Response JSON:")
                    print(response.json())
                else:
                    print("Request failed!")
                    print("Response Body:")
                    print(response.text)

            except requests.exceptions.ConnectionError as e:
                print("\n--- TEST FAILED ---")
                print("Connection Error: Could not connect to the BentoML server.")
                print("Please ensure your BentoML service is running with 'bentoml serve'.")
                print(f"Details: {e}")
import os
if __name__ == "__main__":
    test_event_endpoint()