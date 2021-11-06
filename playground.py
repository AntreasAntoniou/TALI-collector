import pytube


youtube_object = pytube.YouTube(url="https://www.youtube.com/watch?v=7oKjW1OIjuw")
video = youtube_object.streams.get_highest_resolution()
video.download(output_path=f"wikihow_queries/", filename="full_video")
print(youtube_object.length)
