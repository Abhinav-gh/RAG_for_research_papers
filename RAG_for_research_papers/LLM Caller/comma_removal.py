import csv

input_file = "../RAG_app/backend/golden_2.csv"
output_file = "../RAG_app/backend/golden_2_without_commas.csv"

def clean_query(text):
    # Remove all commas inside the query text
    return text.replace(",", "")

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline='', encoding="utf-8") as outfile:

    writer = csv.writer(outfile)

    for line in infile:
        line = line.rstrip("\n")

        # Split only on the LAST comma
        if "," not in line:
            continue  # skip malformed lines

        query, chunk_id = line.rsplit(",", 1)

        cleaned_query = clean_query(query)

        writer.writerow([cleaned_query, chunk_id])

print("✔ Finished. Output written to:", output_file)