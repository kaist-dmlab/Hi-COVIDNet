sed -i -e '$a\ ' README.md
git add *
git commit -m "Update README.md"
git push
git config credential.helper store
git config credential.helper cache
git config credential.helper 'cache --timeout=180000'

