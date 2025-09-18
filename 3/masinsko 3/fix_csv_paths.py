import os
import pandas as pd

novi_prefiks = r"C:\faks\masinsko 3. zadatak\data\IMG"

csv_path = os.path.join("data", "driving_log.csv")
df = pd.read_csv(csv_path, header=None)

for i in range(3):
    df[i] = df[i].apply(lambda path: os.path.join(novi_prefiks, os.path.basename(path.strip())))

clean_path = os.path.join("data", "driving_log_clean.csv")
df.to_csv(clean_path, index=False, header=False)

backup_path = os.path.join("data", "driving_log_backup.csv")
os.rename(csv_path, backup_path)
os.rename(clean_path, csv_path)

print(f"Zamenjene putanje sa prefiksom:\n{novi_prefiks}")
print(f"Originalni fajl sačuvan kao: {backup_path}")
print(f"Prepravljeni fajl preimenovan u: {csv_path}")