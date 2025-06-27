import kagglehub

try:
    path = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq")
    print("Download concluído.")
    print("Path onde foi salvo o dataset:", path)
except Exception as e:
    print(f"Erro no download: {e}")
