#!/bin/sh

cd ~/hugocontent && git add . && git commit -m "changes" && git push origin master && hugo -d ~/hasansi.github.io

cd ~/hasansi.github.io && git add . && git commit -m "changes" && git push origin master
