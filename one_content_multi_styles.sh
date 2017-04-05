#!/bin/bash
SCRIPT=$0
CONTENT_IMAGE=$1
STYLES_DIR=active_styles
CONTENT_INCOMING_DIR=content_incoming
CONTENT_COMPLETE_DIR=content_complete

for i in $( ls $STYLES_DIR); do
    DEST_IMAGE=output/${CONTENT_IMAGE%????}__$i
    if [ ! -f "$DEST_IMAGE" ]; then
      echo attempting to apply $STYLES_DIR/$i to $CONTENT_IMAGE and save into $DEST_IMAGE
      th neural_style.lua -gpu 0 -backend cudnn -style_image $STYLES_DIR/$i -content_image $CONTENT_INCOMING_DIR/$CONTENT_IMAGE -output_image $DEST_IMAGE
    else
      echo "skipping bc $DEST_IMAGE already exists"
    fi
    # python make_gif.py -p ${DEST_IMAGE%????} -r $CONTENT_INCOMING_DIR/$CONTENT_IMAGE -o output/gif/${CONTENT_IMAGE%????}__${i%????}.gif
    cp $DEST_IMAGE output/final/
done
mv $CONTENT_INCOMING_DIR/$CONTENT_IMAGE $CONTENT_COMPLETE_DIR/
