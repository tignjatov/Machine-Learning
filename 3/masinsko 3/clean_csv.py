import csv
import os

data_dir = 'data'
csv_path = os.path.join(os.path.dirname(__file__), data_dir, 'driving_log.csv')
img_dir = os.path.join(data_dir, 'IMG')
clean_csv_path = os.path.join(data_dir, 'driving_log_clean.csv')

with open(csv_path, 'r') as infile, open(clean_csv_path, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    kept = 0
    total = 0
    for row in reader:
        total += 1
        center = os.path.join(img_dir, os.path.basename(row[0].strip()))
        left = os.path.join(img_dir, os.path.basename(row[1].strip()))
        right = os.path.join(img_dir, os.path.basename(row[2].strip()))

        if os.path.exists(center) and os.path.exists(left) and os.path.exists(right):
            writer.writerow(row)
            kept += 1

backup_path = os.path.join(data_dir, 'driving_log_backup.csv')
os.replace(csv_path, backup_path)
os.replace(clean_csv_path, csv_path)

print(f"Zadržano redova: {kept} od ukupno: {total}")
print("Originalni fajl je sačuvan kao:", backup_path)
print("Čisti fajl preimenovan u:", csv_path)