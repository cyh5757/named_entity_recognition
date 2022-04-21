# named_entity_recognition
- 처음부터 해보는 NER

# NER 입력 데이터 
- Sentence, label

# NER 출력 데이터
- Sentence, predict_label, test_label


# model
- PLM(BERT, koBERT(Klue, SKT))
- customize_model

# 수행동작
- data -> pre_process(필요한 데이터 추출(문장, label)) -> Sequence_labeling -> model_train(huggingface_model) -> inference 
- model_train 부분만 customization 수행

# 지금까지 수행
- pre_process, Sequence_labeling


# 해야하는 것
- make model_train

# MODEL 수행 형태
- hugginface's trainer(plm, data_split(train&valid)&shuffle, compute_metrics, hf_trainer)

# .py
tokenizer , model 을 library로 받아오기
def pre_trained->get_datasets -> trainer(huggingface) ->


/*--꿀팁 : 깃 필사할때 main 부터 시작해서, main에따라 def는 나중에 정리 */
