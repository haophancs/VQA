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
# cd ./datasets
# gdown 1Zvd8hi-RjE8Bdvd9kJi6JZ0yOK18fJsz
# unzip -q viclevr.zip && rm viclevr.zip
# cd -

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

python -m coatt.dataset
python -m coatt.dataset_tr
python -m coatt.dataset_va
python -m coatt.dataset_ts
python -m coatt.image_encoding && python -m coatt.main --model coattention --pretrained_embed phow2v.word.300d
