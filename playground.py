import pytube
from rich import print

if __name__ == "__main__":
    video_id = "sgCADNGXdLc"
    yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
    print(yt.streams.all())
    # print interesting attributes
    print(f"Title: {yt.title}")
    print(f"Number of views: {yt.views}")
    print(f"Length of video: {yt.length} seconds")
    print(f"Rating: {yt.rating}")
    print(f"Thumbnail: {yt.thumbnail_url}")
    print(f"Description: {yt.description}")
    print(f"Automated Captions: {yt.captions}")
    # get the highest resolution video stream
    stream = yt.streams.get_highest_resolution()
    # download the video
    stream.download()
    print("Download complete!")
