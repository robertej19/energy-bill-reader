mkdir -p .cache/publaynet
# Download the PublayNet Faster R-CNN weights:
# (If curl/wget errors, try the other; this is the common hosted file used by layoutparser)
curl -L "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1" -o .cache/publaynet/model_final.pth
# or:
# wget -O .cache/publaynet/model_final.pth "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1"

ls -lh .cache/publaynet/model_final.pth   # sanity check, ~330MB
