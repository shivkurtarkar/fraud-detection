docker build -t fdsv . && \
docker run -it \
    -p8000:8000 \
    -eMODEL_FILE=models/best_model.pkl \
    fdsv bash

# -eEXPOSED_PORT=5000 