import pandas as pd

# Baca dataset
df = pd.read_csv("data/dataset.csv")

# Fungsi sederhana buat ngeringkas (ambil 3 kata penting pertama)
def ringkas(teks):
    return " ".join(teks.split()[:3])

# Terapkan ke semua laporan
df["laporan"] = df["laporan"].apply(ringkas)

# Simpan hasilnya
df.to_csv("sortered_Dataset.csv", index=False)

print("Berhasil bikin sortered_Dataset.csv")
