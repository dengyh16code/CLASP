import requests
import base64
import time
import numpy as np
import io

# Load and encode image
with open('../../images/2.jpg', 'rb') as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

start_time = time.time()

# Send request
response = requests.post('http://crane1.d2.comp.nus.edu.sg:55575/process_image', 
                        json={'image': img_base64})
response = response.json()
# Decode the features
features_bytes = base64.b64decode(response['features'])
features_array = np.load(io.BytesIO(features_bytes))

# Decode the dino features
features_dino_bytes = base64.b64decode(response['features_dino'])
features_dino_array = np.load(io.BytesIO(features_dino_bytes))

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

print(features_array.shape)

