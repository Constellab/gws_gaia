
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv .venv --python=python3

# activate vitual env
. ./.venv/bin/activate
#. ./.venv/Scripts/activate

# prism requirement file
python3 -m pip install -r ../gws-py/requirements.txt

# current requirement file
python3 -m pip install -r requirements.txt
