#!/usr/bin/env python3
import re

def fix_float_parameters():
    # Read the file
    with open('crypto_data_analysis_new.py', 'r') as f:
        content = f.read()
    
    # Count original float parameters
    original_floats = re.findall(r'(?:ts_|decay_|correlation|sum|rank|delta|min|max|std|mean)\w*\([^)]*?,\s*\d+\.\d+', content)
    print(f"Found {len(original_floats)} float parameters to fix")
    
    # Pattern to find function calls with float parameters
    # Matches patterns like: function_name(df, ..., 12.345) or function_name(df, ..., 'column', 12.345)
    pattern = r'((?:ts_|decay_|correlation|sum|rank|delta|min|max|std|mean)\w*\([^)]*?,\s*)(\d+\.\d+)(\s*[,)])'
    
    def replace_float(match):
        prefix = match.group(1)
        float_val = match.group(2)
        suffix = match.group(3)
        return f'{prefix}int({float_val}){suffix}'
    
    # Replace all matches
    new_content = re.sub(pattern, replace_float, content)
    
    # Count remaining float parameters
    remaining_floats = re.findall(r'(?:ts_|decay_|correlation|sum|rank|delta|min|max|std|mean)\w*\([^)]*?,\s*\d+\.\d+', new_content)
    print(f"Remaining float parameters: {len(remaining_floats)}")
    
    if len(remaining_floats) < len(original_floats):
        # Write back the fixed content
        with open('crypto_data_analysis_new.py', 'w') as f:
            f.write(new_content)
        print(f"Fixed {len(original_floats) - len(remaining_floats)} float parameters")
    else:
        print("No changes made")

if __name__ == "__main__":
    fix_float_parameters()
