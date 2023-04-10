pip install -r requirements.txt

mkdir -p /content/VQA/datasets
mkdir -p /content/VQA/pretrained
mkdir -p /content/VQA/outputs

mkdir -p /content/VQA/outputs/coatt/tr_enc
mkdir -p /content/VQA/outputs/coatt/va_enc
mkdir -p /content/VQA/outputs/coatt/ts_enc
mkdir -p /content/VQA/outputs/coatt/chk_coattention
mkdir -p /content/VQA/outputs/coatt/log

# Download ViVQA datasets
cd /content/VQA/datasets
gdown --id 1Yc36OjdwpXt14eOJ6HjK78prV-eT8Y83
unzip -q vivqa.zip && rm vivqa.zip
cd /content/VQA

# Download pretrained word vectors
cd /content/VQA/pretrained
wget https://public.vinai.io/word2vec_vi_words_100dims.zip
unzip -q word2vec_vi_words_100dims.zip
rm word2vec_vi_words_100dims.zip

wget https://public.vinai.io/word2vec_vi_words_300dims.zip
unzip -q word2vec_vi_words_300dims.zip
rm word2vec_vi_words_300dims.zip

wget https://public.vinai.io/word2vec_vi_syllables_100dims.zip
unzip -q word2vec_vi_syllables_100dims.zip
rm word2vec_vi_syllables_100dims.zip

wget https://public.vinai.io/word2vec_vi_syllables_300dims.zip
unzip -q word2vec_vi_syllables_300dims.zip
rm word2vec_vi_syllables_300dims.zip
cd /content/VQA
