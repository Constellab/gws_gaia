
python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv .venv --python=python3

# activate vitual env
. ./.venv/bin/activate
#. ./.venv/Scripts/activate

# prism requirement file
python -m pip install -r ../gws-py/requirements.txt

# current requirement file
python -m pip install -r requirements.txt
