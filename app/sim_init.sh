# This script downloads stuff, about 12.5GB

# Get GloVe and fastText vectors
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip  # 2075M
unzip GloVe/glove.840B.300d.zip -d GloVe/  # 5.65 GB
mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip # 1453M
unzip fastText/crawl-300d-2M.vec.zip -d fastText/ # 4.51GB

# Download the sentence encoder
mkdir encoder
curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl  # 154MB
curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl  # 154MB
