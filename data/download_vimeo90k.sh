#!/bin/bash
# download_vimeo90k_only.sh

DATASET_DIR="./datasets/vimeo90k"
mkdir -p $DATASET_DIR

echo "Downloading Vimeo-90K septuplet dataset (~82GB)..."
echo "This will take several hours. Use screen/tmux to keep it running."
echo ""

wget -c http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip -P $DATASET_DIR

# Check if download completed
if [ -f "$DATASET_DIR/vimeo_septuplet.zip" ]; then
    FILE_SIZE=$(du -h "$DATASET_DIR/vimeo_septuplet.zip" | cut -f1)
    echo ""
    echo "Download complete! File size: $FILE_SIZE"
    echo "Location: $DATASET_DIR/vimeo_septuplet.zip"
    echo ""
    echo "Next steps:"
    echo "1. Verify file integrity (optional): md5sum $DATASET_DIR/vimeo_septuplet.zip"
    echo "2. Extract manually: unzip $DATASET_DIR/vimeo_septuplet.zip -d $DATASET_DIR"
    echo "3. Delete zip after successful extraction: rm $DATASET_DIR/vimeo_septuplet.zip"
else
    echo "Download failed or incomplete."
    exit 1
fi