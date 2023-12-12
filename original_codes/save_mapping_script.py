import json

# Your label_mapping dictionary
label_mapping = {
    'Accent Label: Midland': 9,
    'Accent Label: InlandNorth': 3,
    'Accent Label: MidAtlantic': 5,
    'Accent Label: Southern': 8,
    'Accent Label: NYC': 6,
    'Accent Label: WesternNE': 12,
    'Accent Label: Western': 11,
    'Accent Label: EasternNE': 1,
    'Accent Label: InlandSouth': 4,
    'Accent Label: Northern': 7,
    'Accent Label: WestPennsylvania': 10,
    'Accent Label: Florida': 2,
}

# Save to a JSON file
with open('label_mapping.json', 'w') as json_file:
    json.dump(label_mapping, json_file)

