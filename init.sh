pip install -r requirements.txt

mkdir -p ./datasets
mkdir -p ./pretrained
mkdir -p ./outputs

mkdir -p ./outputs/coatt/tr_enc
mkdir -p ./outputs/coatt/va_enc
mkdir -p ./outputs/coatt/ts_enc
mkdir -p ./outputs/coatt/chk_coattention
mkdir -p ./outputs/coatt/log

# Download ViVQA datasets
cd ./datasets
gdown 1Yc36OjdwpXt14eOJ6HjK78prV-eT8Y83
unzip -q vivqa.zip && rm vivqa.zip
cd -

# Download pretrained word vectors
cd ./pretrained
# wget https://public.vinai.io/word2vec_vi_words_100dims.zip
# unzip -q word2vec_vi_words_100dims.zip
# rm word2vec_vi_words_100dims.zip

wget https://public.vinai.io/word2vec_vi_words_300dims.zip
unzip -q word2vec_vi_words_300dims.zip
rm word2vec_vi_words_300dims.zip

# wget https://public.vinai.io/word2vec_vi_syllables_100dims.zip
# unzip -q word2vec_vi_syllables_100dims.zip
# rm word2vec_vi_syllables_100dims.zip

# wget https://public.vinai.io/word2vec_vi_syllables_300dims.zip
# unzip -q word2vec_vi_syllables_300dims.zip
# rm word2vec_vi_syllables_300dims.zip
cd -
