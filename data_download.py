import requests, argparse, os, sys
BASE_DIR = r"D:\crime_project2"
os.makedirs(BASE_DIR, exist_ok=True)

def download_csv(output, limit):
    url = f"https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit={limit}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output, "wb") as f:
            for chunk in r.iter_content(chunk_size=512*1024):
                if chunk:
                    f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "chicago_crimes.csv"))
    parser.add_argument("--limit", default=500000, type=int)
    args = parser.parse_args()
    print("Downloading to:", args.output)
    try:
        download_csv(args.output, args.limit)
        print("Downloaded:", args.output)
    except Exception as e:
        print("Download failed:", e, file=sys.stderr)
        raise
