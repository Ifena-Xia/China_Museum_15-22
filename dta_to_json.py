import pandas as pd

# Replace with your actual file path
file_path = "...cepsw1parentEN.dta"

# Load the dataset
df = pd.read_stata(file_path, convert_categoricals=False)

# Check the coloum names
print(df.columns.tolist())

# Select only the columns you need
selected_columns = ['ids', 'clsids', 'schids', 'ctyids', 'ba1705', 'be07', 'be29', 'grade9']
df_selected = df[selected_columns]

# Add survey year manually
df_selected['survey_year'] = '2013-2014'

# Convert to a list of dictionaries (records format)
json_data = df_selected.to_dict(orient='records')

# Save to a JSON file
import json
with open("ceps_selected.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

# Print a single JSON entry
import pprint
pprint.pprint(json_data[0])
