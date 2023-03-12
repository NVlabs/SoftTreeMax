FROM nvcr.io/nvidian/pytorch:20.12-py3 as base
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install ffmpeg libsm6 libxext6 libxrender-dev -y
RUN pip install atari_py
RUN pip install wandb plotly
RUN git clone --recursive https://github.com/NVLabs/cule -b bfs
RUN cd cule && python setup.py install && cd ..
RUN git clone https://github.com/nvlabs/softtreemax.git
RUN pip install stable-baselines3==1.4.0 -U --no-deps
RUN pip uninstall -y importlib_metadata && pip install importlib_metadata -U
RUN pip3 install --upgrade requests
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN mkdir /workspace/ROM/
RUN apt-get install unrar
RUN unrar e -y /workspace/Roms.rar /workspace/ROM/
RUN python -m atari_py.import_roms /workspace/ROM/
