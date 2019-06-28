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

# 先安裝 pipenv, tensorflow, matplotlib
RUN pip install pipenv numpy tensorflow matplotlib
RUN pipenv install --three

# pipenv shell

RUN cd /workspace && git clone https://github.com/libgirlenterprise/lgl.git
# 修改程式 從 conda 改成 pipenv
RUN sed -i 's/conda/pipenv/g' lgl/lgl/get_unresolve.py
RUN echo `tail -n 1 lgl/lgl/get_unresolve.py`

# 包成 lib, 讓我可以用 lgl install
RUN cd lgl && pip install .
RUN echo `lgl --help`

# 把 demo 檔案移進 libgirl
RUN cd /workspace && lgl init libgirl
COPY demo_TF.py /workspace/libgirl/libgirl/libgirl.py

# 跑第一次試試看
RUN cd libgirl && lgl install && lgl run libgirl

CMD ["/bin/bash"]
