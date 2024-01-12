# OUTPUT=../../../data/indexes/bm25
# INPUT=../../../data/bm25_collection
# INPUT=/home/wangym/data1/dataset/topiocqa/bm25_collection
# OUTPUT=/home/wangym/data1/dataset/topiocqa/indexes/bm25
INPUT=/home/wangym/data1/dataset/qrecc/bm25_collection
OUTPUT=/home/wangym/data1/dataset/qrecc/indexes/bm25

if [ ! -f "$OUTPUT" ]; then
    echo "Creating index..."
    python -m pyserini.index -collection JsonCollection \
                            -generator DefaultLuceneDocumentGenerator \
                            -threads 20 \
                            -input ${INPUT} \
                            -index ${OUTPUT} \
							-storePositions -storeDocvectors -storeRaw
fi
