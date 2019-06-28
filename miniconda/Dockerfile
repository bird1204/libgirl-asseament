# 20190625 libgirl pre-test-01
# 在 docker 裡裝 conda, 並且使用 new conda env 安裝 lgl
# 詳細步驟
# - 安裝Linux (建議使用虛擬機器或docker, 但因docker設定較為複雜, 可依您對其熟悉程度自由選用)
# - 於Linux 環境中安裝Conda
# - 於conda創建新環境並安裝lgl, 安裝方式請詳: lgl on GitHub (lgl為本團隊為python開發的Launcher開源專案)
# https://github.com/libgirlenterprise/lgl.git
# docker build -t bird1204/libgirl-pretest-miniconda miniconda
# docker run -i -t bird1204/libgirl-pretest-miniconda

# 直接使用官方的 image
FROM continuumio/miniconda:latest
RUN conda update -y conda

RUN mkdir workspace
WORKDIR workspace
COPY demo_TF.py demo_TF.py 

# 安裝 conda 需要用到的 lib
RUN conda install -y python=3.7 git pip tensorflow matplotlib
RUN echo `conda --version`

# 建立 conda 環境，並且指定要用哪些 lib
# 因為要用到的套件很少就直接寫在裡面，數量多的話可以用 environment.yml
# COPY environment.yml ./
RUN conda create -n libgirl python git pip tensorflow matplotlib

# 進入虛擬環境，安裝 lgl
RUN conda run -n libgirl && \
    pip install lgl tensorflow matplotlib && \ 
    echo `lgl --help`

# 把 demo 檔案移進 libgirl
RUN cd /workspace && lgl init libgirl
RUN mv -f /workspace/demo_TF.py /workspace/libgirl/libgirl/libgirl.py

# 跑第一次試試看
RUN cd libgirl && lgl install && lgl run libgirl

CMD ["/bin/bash"]

