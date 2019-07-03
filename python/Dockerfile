# 建立用 pipenv 跑 lgl 的環境
# - 針對  python 技術實作測驗, 在此提供您題目為: 
# - 修改lgl的程式碼 (使用git clone), 將lgl run (包含lgl install) 從使用conda改成使用pipenv, 並經測試 (可擇任意python專案)確定改後可正常運作.
# https://github.com/libgirlenterprise/lgl.git
# docker build -t bird1204/libgirl-pretest-python python
# docker run -i -t bird1204/libgirl-pretest-python

FROM python:latest

RUN apt-get update && apt-get install -y \
    python-pip git vim

RUN mkdir workspace
WORKDIR workspace

# 把檔案移進 docker
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

# 下載 lgl
RUN cd /workspace && git clone https://github.com/libgirlenterprise/lgl.git
# 修改程式 從 conda 改成 pipenv
RUN sed -i 's/conda/pipenv/g' lgl/lgl/get_unresolve.py
RUN echo `tail -n 1 lgl/lgl/get_unresolve.py`

# 安裝 pipenv, 並根據已經準備好的 Pipfile, Pipfile.lock 安裝套件
RUN pip install pipenv
RUN pipenv --python 3.7 && pipenv sync

# 在 pipenv 裡包成 lib, 讓我可以用 lgl install
RUN pipenv install -e ./lgl
RUN echo `pipenv run lgl --help`

# 起始 libgirl, 把 demo 檔案移進 libgirl
RUN pipenv run lgl init libgirl
COPY demo_TF.py /workspace/libgirl/libgirl/libgirl.py

# 在 pipenv 跑第一次 install, run 試試看
RUN pipenv run lgl --verbose install /workspace/libgirl/libgirl/libgirl.py 
RUN cd /workspace/libgirl && pipenv run lgl --verbose run libgirl

CMD ["/bin/bash"]


# user 不會想要 pipenv run lgl, 要直接 run lgl
# 有 Pipfile, 就不會 run lg install 了