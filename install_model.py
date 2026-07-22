from sentence_transformers import SentenceTransformer

print("[*] Initializing model installation script...")
print("[*] Downloading 'all-MiniLM-L6-v2' from Hugging Face hub...")
print("[*] Note: This will download roughly 120MB of data. Please wait...")

# This line automatically triggers the download if it's not already cached locally
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\n[+] Success! Model downloaded and loaded into memory.")
print(f"[+] Model Vector Space Dimensions: 384")

# Test a simple inference matrix update
test_vector = model.encode("Python MLOps pipeline development.")
print(f"[+] Verified Local Inference. Test Vector Preview (First 3 dimensions): {test_vector[:3]}")
