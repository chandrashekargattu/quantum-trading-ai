#!/usr/bin/env python3
"""Fix all regex= to pattern= for Pydantic v2 compatibility"""
import os
import re

def fix_regex_in_file(filepath):
    """Replace regex= with pattern= in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace regex= with pattern=
    new_content = re.sub(r'\bregex=', 'pattern=', content)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"✅ Fixed: {filepath}")
        return True
    return False

# Fix all schema files
schemas_dir = "app/schemas"
fixed_count = 0

for filename in os.listdir(schemas_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(schemas_dir, filename)
        if fix_regex_in_file(filepath):
            fixed_count += 1

print(f"\n🎯 Total files fixed: {fixed_count}")
print("✨ All regex= occurrences have been replaced with pattern=")
