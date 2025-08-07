#!/usr/bin/env python3
"""
Script to create a JSON file that will cause UTF-8 encoding errors
when loaded with Python's json module
"""

import json

def create_problematic_json():
    """
    Create a JSON file with invalid UTF-8 bytes that will cause json.load() to fail
    """
    
    # First, create a valid JSON structure
    valid_json_data = {
        "name": "Test Document",
        "description": "This JSON contains invalid UTF-8 bytes",
        "data": {
            "numbers": [1, 2, 3, 4, 5],
            "text": "Some valid text here",
            "nested": {
                "key1": "value1",
                "key2": "value2"
            }
        },
        "status": "problematic"
    }
    
    # Convert to JSON string
    json_string = json.dumps(valid_json_data, indent=2)
    
    # Write the file in binary mode so we can inject invalid UTF-8 bytes
    with open('problematic.json', 'wb') as f:
        # Write most of the JSON as valid UTF-8
        f.write(json_string[:200].encode('utf-8'))
        
        # Insert invalid UTF-8 byte sequences
        # These bytes are invalid as UTF-8:
        # 0xC0 0x80 - overlong encoding (invalid)
        # 0xFE - invalid start byte
        # 0xFF - invalid start byte
        # 0xC2 alone - incomplete sequence
        f.write(b'\xC0\x80')  # Overlong encoding
        
        # Continue with more valid JSON
        f.write(json_string[200:400].encode('utf-8'))
        
        # Insert more problematic bytes
        f.write(b'\xFE\xFF')  # Invalid start bytes
        
        # Add the rest of the JSON
        f.write(json_string[400:].encode('utf-8'))
        
        # Add a lone high surrogate (invalid)
        f.write(b'\xED\xA0\x80')  # UTF-8 encoding of U+D800 (high surrogate)

def create_alternative_problematic_json():
    """
    Alternative method: Create JSON with invalid UTF-8 in string values
    """
    
    # Create a JSON-like structure manually with embedded invalid bytes
    json_like_content = b'''{
  "title": "Document with encoding issues",
  "content": "This text contains invalid UTF-8: \xC0\x80\xFE\xFF",
  "data": {
    "field1": "Normal text",
    "field2": "Text with problems: \xED\xA0\x80\xC2",
    "numbers": [1, 2, 3]
  },
  "status": "invalid_encoding"
}'''
    
    with open('problematic_alternative.json', 'wb') as f:
        f.write(json_like_content)

def demonstrate_error():
    """
    Demonstrate how the problematic JSON files cause errors
    """
    
    files_to_test = ['problematic.json', 'problematic_alternative.json']
    
    for filename in files_to_test:
        print(f"\n--- Testing {filename} ---")
        
        try:
            # This will fail with UnicodeDecodeError
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Successfully loaded {filename}")
            
        except UnicodeDecodeError as e:
            print(f"✗ UnicodeDecodeError when loading {filename}:")
            print(f"  {e}")
            
        except json.JSONDecodeError as e:
            print(f"✗ JSONDecodeError when loading {filename}:")
            print(f"  {e}")
            
        # Try reading as bytes to show the raw content
        try:
            with open(filename, 'rb') as f:
                raw_bytes = f.read()
            print(f"  File size: {len(raw_bytes)} bytes")
            print(f"  First 100 bytes: {raw_bytes[:100]}")
            
        except Exception as e:
            print(f"  Could not read raw bytes: {e}")

if __name__ == "__main__":
    print("Creating problematic JSON files...")
    
    # Create the problematic files
    create_problematic_json()
    create_alternative_problematic_json()
    
    print("Created files:")
    print("  - problematic.json")
    print("  - problematic_alternative.json")
    
    print("\nTo trigger the UTF-8 error, try:")
    print("import json")
    print("with open('problematic.json', 'r', encoding='utf-8') as f:")
    print("    data = json.load(f)")
    
    print("\nDemonstrating the errors:")
    demonstrate_error()
    
    print("\nAlternative ways to handle these files:")
    print("1. Read with error handling:")
    print("   with open('problematic.json', 'r', encoding='utf-8', errors='replace') as f:")
    print("       # This will replace invalid bytes with � characters")
    print("2. Read as bytes first:")
    print("   with open('problematic.json', 'rb') as f:")
    print("       raw_data = f.read()")
    print("       # Then handle encoding issues manually")
