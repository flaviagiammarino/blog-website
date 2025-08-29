python3 -m pip install ./_ablog
rm -r docs
make html
git add --all
git commit -m "update website"
git push origin main