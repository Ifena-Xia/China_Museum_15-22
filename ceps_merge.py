import json
from pathlib import Path

# 1) Define paths
desktop = Path.home() / "Desktop"
path_s   = desktop / "ceps_s_transformed.json"
path_p   = desktop / "ceps_p_transformed.json"
out_path = desktop / "ceps_merge.json"

# 2) Load the two JSON files
with path_s.open("r", encoding="utf-8") as f:
    s_records = json.load(f)

with path_p.open("r", encoding="utf-8") as f:
    p_records = json.load(f)

# 3) Build a lookup dict for the 's' records
#    keyed by the composite (student_id, class_id, school_id, county_id)
s_map = {
    (r["student_id"], r["class_id"], r["school_id"], r["county_id"]): r
    for r in s_records
}
# For each record r in the list s_records(student_info)
# For each student record r, we pull out the four ID fields and pack them into a Python tuple
# This tuple uniquely identifies that student’s “row”
# The s_map looks like
# {
#  (1, 1, 1, 1): { "student_id":1, "class_id":1, … },
#  (1, 1, 1, 2): { … },
#  …
#}

# 4) Merge
merged = [] # initializes an empty list
for p in p_records:
    key = (p["student_id"], p["class_id"], p["school_id"], p["county_id"])
# .get(key) attempts to fetch the student record s that has the same key
    s = s_map.get(key)
    if s:
        # combine: start from s, then overlay p
        merged_rec = {**s, **p}
        merged.append(merged_rec)
    # else: you could choose to keep unmatched p or log a warning
# {**s, **p} uses dict unpacking to create a new dict containing all key–value pairs from s, then all from p
#If s and p share the same field name, the value from p will overwrite the one from s

# 5) Write out the merged JSON
with out_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(merged)} records to {out_path}")
