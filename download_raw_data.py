import requests
import os

def download_osf_file(osf_url, filename="all_sctt_jrt.csv"):
    """Simple function to download files from OSF"""
    
    # Extract file ID from OSF URL
    file_id = osf_url.split("/files/")[1].split("?")[0]
    
    # Get view_only parameter if present
    view_only = ""
    if "view_only=" in osf_url:
        view_only = f"?view_only={osf_url.split('view_only=')[1].split('&')[0]}"
    
    # Construct download URL
    download_url = f"https://osf.io/download/{file_id}/{view_only}"
    
    # Download the file
    response = requests.get(download_url)
    response.raise_for_status()
    
    # Save to current directory
    with open(filename, "wb") as f:
        f.write(response.content)
    
    print(f"Downloaded: {filename}")
    return filename

# Download the dataset
# Does not include the "cut/keep" data column.
# url = "https://osf.io/439zs/files/wezak?view_only=4cbda208526948a99afba0050a2c043f"

url = "https://osf.io/439zs/files/zy3cv?view_only=4cbda208526948a99afba0050a2c043f"
download_osf_file(url, "all_sctt_jrt.csv")
