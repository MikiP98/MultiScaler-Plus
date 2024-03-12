@cd src
git submodule update --init --recursive
pip install --no-dependencies transformers==4.31.0
pip install -r requirements.txt
@PAUSE