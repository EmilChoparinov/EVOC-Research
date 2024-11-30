# Using FFMPEG we convert the mov file that
for file in *.mov; do
    ffmpeg -i "$file" -c:v libx264 -crf 23 -preset veryfast -c:a aac -b:a 128k "${file%.mov}.mp4"
done
