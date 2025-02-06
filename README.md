# PotatoDiseaseClassification

Command to start tf-serving

docker run -t --rm -p 8501:8501 \
  -v /home/vadim/Desktop/prog/ML/Projects/PotatoDiseaseClassification/models:/models \
  -e MODEL_NAME=potato_disease_classifier1 \
  tensorflow/serving:latest \
  --rest_api_port=8501


gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project potato-disiase-classification