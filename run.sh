#git clone https://github.com/flaviagiammarino/ablog
#python3 -m pip install ./ablog
#chmod -R +w ablog
#rm -r ablog
rm -r docs
make html
git add --all
git commit -m "update website"
git push origin main