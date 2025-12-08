import os
from transformers.convert_slow_tokenizer import import_protobuf
import sentencepiece as spm

# --- Start of user configuration ---

# 1. Path to your original SentencePiece model
original_model_path = "/home/willw2/expressive_s2st/index-tts/checkpoints/bpe.model"  # <--- CHANGE THIS

# 2. The single token you want to add (e.g., a special or control token)
stress_token_start = "<*>"  # <--- CHANGE THIS
stress_token_close = "</*>"

# 3. Path to save the new, extended model
new_model_path = "/home/willw2/expressive_s2st/index-tts/checkpoints/bpe_extended.model"  # <--- CHANGE THIS

# --- End of user configuration ---

# Load the protobuf library
model_pb2 = import_protobuf()

# Create a new model proto object
m = model_pb2.ModelProto()

# Read and parse the original model file
try:
    with open(original_model_path, "rb") as f:
        m.ParseFromString(f.read())
except FileNotFoundError:
    print(f"Error: Original model not found at '{original_model_path}'")
    print("Please update the 'original_model_path' variable in the script.")
    exit()


print("--- Original Model ---")
print(f"Vocabulary size: {len(m.pieces)}")

# Check if the new tokens already exist
existing_pieces = {p.piece for p in m.pieces}
tokens_to_add = [stress_token_start, stress_token_close]
existing_tokens = [token for token in tokens_to_add if token in existing_pieces]

if existing_tokens:
    print(f"\nTokens {existing_tokens} already exist in the model. Aborting.")
else:
    # Add both tokens to the model
    for token in tokens_to_add:
        # Create a new SentencePiece piece
        new_piece = model_pb2.ModelProto.SentencePiece()
        new_piece.piece = token
        new_piece.score = 0.0  # Scores are typically 0.0 for special/control tokens
        new_piece.type = 4     # 4 corresponds to the CONTROL type for special tokens

        # Add the new piece to the model's vocabulary
        m.pieces.append(new_piece)

    print("\n--- After Modification ---")
    print(f"Successfully added tokens: {tokens_to_add}")
    print(f"New vocabulary size: {len(m.pieces)}")

    # Write the modified model to a new file
    with open(new_model_path, "wb") as f:
        f.write(m.SerializeToString())
    print(f"New model saved to: {new_model_path}")

    # --- Verification Step ---
    print("\n--- Testing New Model ---")
    sp = spm.SentencePieceProcessor()
    sp.load(new_model_path)

    # Test both tokens
    for token in tokens_to_add:
        token_id = sp.piece_to_id(token)
        print(f"ID for token '{token}': {token_id}")

    test_sentence = f"This is a test with special stress open token {stress_token_start}, and special stress close token {stress_token_close}."
    encoded_ids = sp.encode_as_ids(test_sentence)
    print(f"Test sentence encoded as IDs: {encoded_ids}")
    print("Verification complete.")