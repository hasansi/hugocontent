#!/bin/sh

# If a command fails then the deploy stops
set -e

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Build the project.
#hugo -t coder

# Go To hugo folder
cd ~/hugocontent

# Add changes to git.
git add .

# Commit changes.
msg="rebuilding site $(date)"
if [ -n "$*" ]; then
		msg="$*"
	fi
	git commit -m "$msg"

	# Push source and build repos.
	git push origin master


hugo -d ~/hasansi.github.io

	
# Go To username folder
cd ~/hasansi.github.io && git add . && git commit -m "changes" && git push origin master

