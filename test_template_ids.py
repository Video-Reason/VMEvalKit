#!/usr/bin/env python3
"""Test the template ID generation to ensure no conflicts."""

import requests
from bs4 import BeautifulSoup

# Fetch the main page
response = requests.get("http://localhost:5000")
if response.status_code != 200:
    print(f"Error: Failed to fetch page (status {response.status_code})")
    exit(1)

# Parse the HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Find all model sections
model_sections = soup.find_all('section', class_='model-section')
print(f"Found {len(model_sections)} model sections")

# Check for ID conflicts
all_ids = set()
conflicts = []

# Check model IDs
for i, section in enumerate(model_sections, 1):
    model_id = f"model-{i}"
    model_content = soup.find(id=model_id)
    if model_content:
        if model_id in all_ids:
            conflicts.append(f"Duplicate ID: {model_id}")
        all_ids.add(model_id)
        print(f"✓ Model {i} ID: {model_id}")
    else:
        print(f"✗ Model {i} ID not found: {model_id}")
    
    # Check domain IDs within this model
    domain_headers = section.find_all('div', class_='domain-header')
    for j, domain in enumerate(domain_headers, 1):
        domain_id = f"domain-{i}-{j}"
        domain_content = soup.find(id=domain_id)
        if domain_content:
            if domain_id in all_ids:
                conflicts.append(f"Duplicate ID: {domain_id}")
            all_ids.add(domain_id)
            print(f"  ✓ Domain {j} ID: {domain_id}")
        else:
            print(f"  ✗ Domain {j} ID not found: {domain_id}")

print("\n" + "=" * 50)
if conflicts:
    print("❌ ID Conflicts Found:")
    for conflict in conflicts:
        print(f"  - {conflict}")
else:
    print("✅ No ID conflicts found!")

# Check auto-expand status
print("\n" + "=" * 50)
print("Checking auto-expand status:")

# First model should be expanded
first_model = soup.find(id="model-1")
if first_model:
    style = first_model.get('style', '')
    if 'display: block' in style or 'display:block' in style:
        print("✓ First model is expanded")
    else:
        print("✗ First model is NOT expanded (should be)")
else:
    print("✗ First model not found")

# First domain of first model should be expanded  
first_domain = soup.find(id="domain-1-1")
if first_domain:
    style = first_domain.get('style', '')
    if 'display: block' in style or 'display:block' in style:
        print("✓ First domain of first model is expanded")
    else:
        print("✗ First domain is NOT expanded (should be)")
else:
    print("✗ First domain not found")

print("\n✅ Template test complete!")
