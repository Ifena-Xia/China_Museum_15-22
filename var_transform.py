import json

# Define the mapping from old keys to new keys.
mapping = {
    "ids": "student_id",
    "clsids": "class_id",
    "schids": "school_id",
    "ctyids": "county_id",
    "grade9": "grade9",
    "ba1705": "frequency",
    "be07": "parent_edu",
    "be29": "living_area"
}

# Specify the input and output file paths.
# Update the paths if necessary for your environment.
input_path = "...ceps_selected.json"
output_path = "...ceps.json"

# Load the JSON data from the input file.
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each record and build a new record with renamed keys.
new_data = []
for record in data:
    new_record = {}
    for old_key, new_key in mapping.items():
        if old_key in record:
            new_record[new_key] = record[old_key]
    # Optionally, include any keys that aren't being renamed.
    # For example, to copy over all other fields, uncomment the following:
    # for key in record:
    #     if key not in mapping:
    #         new_record[key] = record[key]
    new_data.append(new_record)

# Write the transformed data to the output file.
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("Transformation complete. Transformed file saved at:", output_path)
